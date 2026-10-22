# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""Parses the mmCIF file format."""
import collections
import dataclasses
import io
import logging
from typing import Any, Mapping, Optional, Sequence, Tuple
import re
import string

from Bio import PDB
from Bio.Data import PDBData
from Bio.PDB import PDBParser
import numpy as np
import modelcif
import modelcif.model
import modelcif.dumper
import modelcif.reference
import modelcif.protocol
import modelcif.alignment
import modelcif.qa_metric

from minifold.utils import residue_constants


# Type aliases:
ChainId = str
PdbHeader = Mapping[str, Any]
PdbStructure = PDB.Structure.Structure
SeqRes = str
MmCIFDict = Mapping[str, Sequence[str]]


@dataclasses.dataclass(frozen=True)
class Monomer:
    id: str
    num: int


# Note - mmCIF format provides no guarantees on the type of author-assigned
# sequence numbers. They need not be integers.
@dataclasses.dataclass(frozen=True)
class AtomSite:
    residue_name: str
    author_chain_id: str
    mmcif_chain_id: str
    author_seq_num: str
    mmcif_seq_num: int
    insertion_code: str
    hetatm_atom: str
    model_num: int


# Used to map SEQRES index to a residue in the structure.
@dataclasses.dataclass(frozen=True)
class ResiduePosition:
    chain_id: str
    residue_number: int
    insertion_code: str


@dataclasses.dataclass(frozen=True)
class ResidueAtPosition:
    position: Optional[ResiduePosition]
    name: str
    is_missing: bool
    hetflag: str


@dataclasses.dataclass(frozen=True)
class MmcifObject:
    """Representation of a parsed mmCIF file.

    Contains:
      file_id: A meaningful name, e.g. a pdb_id. Should be unique amongst all
        files being processed.
      header: Biopython header.
      structure: Biopython structure.
      chain_to_seqres: Dict mapping chain_id to 1 letter amino acid sequence. E.g.
        {'A': 'ABCDEFG'}
      seqres_to_structure: Dict; for each chain_id contains a mapping between
        SEQRES index and a ResidueAtPosition. e.g. {'A': {0: ResidueAtPosition,
                                                          1: ResidueAtPosition,
                                                          ...}}
      raw_string: The raw string used to construct the MmcifObject.
    """

    file_id: str
    header: PdbHeader
    structure: PdbStructure
    chain_to_seqres: Mapping[ChainId, SeqRes]
    seqres_to_structure: Mapping[ChainId, Mapping[int, ResidueAtPosition]]
    raw_string: Any


@dataclasses.dataclass(frozen=True)
class ParsingResult:
    """Returned by the parse function.

    Contains:
      mmcif_object: A MmcifObject, may be None if no chain could be successfully
        parsed.
      errors: A dict mapping (file_id, chain_id) to any exception generated.
    """

    mmcif_object: Optional[MmcifObject]
    errors: Mapping[Tuple[str, str], Any]


class ParseError(Exception):
    """An error indicating that an mmCIF file could not be parsed."""


def mmcif_loop_to_list(
    prefix: str, parsed_info: MmCIFDict
) -> Sequence[Mapping[str, str]]:
    """Extracts loop associated with a prefix from mmCIF data as a list.

    Reference for loop_ in mmCIF:
      http://mmcif.wwpdb.org/docs/tutorials/mechanics/pdbx-mmcif-syntax.html

    Args:
      prefix: Prefix shared by each of the data items in the loop.
        e.g. '_entity_poly_seq.', where the data items are _entity_poly_seq.num,
        _entity_poly_seq.mon_id. Should include the trailing period.
      parsed_info: A dict of parsed mmCIF data, e.g. _mmcif_dict from a Biopython
        parser.

    Returns:
      Returns a list of dicts; each dict represents 1 entry from an mmCIF loop.
    """
    cols = []
    data = []
    for key, value in parsed_info.items():
        if key.startswith(prefix):
            cols.append(key)
            data.append(value)

    assert all([len(xs) == len(data[0]) for xs in data]), (
        "mmCIF error: Not all loops are the same length: %s" % cols
    )

    return [dict(zip(cols, xs)) for xs in zip(*data)]


def mmcif_loop_to_dict(
    prefix: str,
    index: str,
    parsed_info: MmCIFDict,
) -> Mapping[str, Mapping[str, str]]:
    """Extracts loop associated with a prefix from mmCIF data as a dictionary.

    Args:
      prefix: Prefix shared by each of the data items in the loop.
        e.g. '_entity_poly_seq.', where the data items are _entity_poly_seq.num,
        _entity_poly_seq.mon_id. Should include the trailing period.
      index: Which item of loop data should serve as the key.
      parsed_info: A dict of parsed mmCIF data, e.g. _mmcif_dict from a Biopython
        parser.

    Returns:
      Returns a dict of dicts; each dict represents 1 entry from an mmCIF loop,
      indexed by the index column.
    """
    entries = mmcif_loop_to_list(prefix, parsed_info)
    return {entry[index]: entry for entry in entries}


def parse(
    *, file_id: str, mmcif_string: str, catch_all_errors: bool = True
) -> ParsingResult:
    """Entry point, parses an mmcif_string.

    Args:
      file_id: A string identifier for this file. Should be unique within the
        collection of files being processed.
      mmcif_string: Contents of an mmCIF file.
      catch_all_errors: If True, all exceptions are caught and error messages are
        returned as part of the ParsingResult. If False exceptions will be allowed
        to propagate.

    Returns:
      A ParsingResult.
    """
    errors = {}
    try:
        parser = PDB.MMCIFParser(QUIET=True)
        handle = io.StringIO(mmcif_string)
        full_structure = parser.get_structure("", handle)
        first_model_structure = _get_first_model(full_structure)
        # Extract the _mmcif_dict from the parser, which contains useful fields not
        # reflected in the Biopython structure.
        parsed_info = parser._mmcif_dict  # pylint:disable=protected-access

        # Ensure all values are lists, even if singletons.
        for key, value in parsed_info.items():
            if not isinstance(value, list):
                parsed_info[key] = [value]

        header = _get_header(parsed_info)

        # Determine the protein chains, and their start numbers according to the
        # internal mmCIF numbering scheme (likely but not guaranteed to be 1).
        valid_chains = _get_protein_chains(parsed_info=parsed_info)
        if not valid_chains:
            return ParsingResult(
                None, {(file_id, ""): "No protein chains found in this file."}
            )
        seq_start_num = {
            chain_id: min([monomer.num for monomer in seq])
            for chain_id, seq in valid_chains.items()
        }

        # Loop over the atoms for which we have coordinates. Populate two mappings:
        # -mmcif_to_author_chain_id (maps internal mmCIF chain ids to chain ids used
        # the authors / Biopython).
        # -seq_to_structure_mappings (maps idx into sequence to ResidueAtPosition).
        mmcif_to_author_chain_id = {}
        seq_to_structure_mappings = {}
        for atom in _get_atom_site_list(parsed_info):
            if atom.model_num != "1":
                # We only process the first model at the moment.
                continue

            mmcif_to_author_chain_id[atom.mmcif_chain_id] = atom.author_chain_id

            if atom.mmcif_chain_id in valid_chains:
                hetflag = " "
                if atom.hetatm_atom == "HETATM":
                    # Water atoms are assigned a special hetflag of W in Biopython. We
                    # need to do the same, so that this hetflag can be used to fetch
                    # a residue from the Biopython structure by id.
                    if atom.residue_name in ("HOH", "WAT"):
                        hetflag = "W"
                    else:
                        hetflag = "H_" + atom.residue_name
                insertion_code = atom.insertion_code
                if not _is_set(atom.insertion_code):
                    insertion_code = " "
                position = ResiduePosition(
                    chain_id=atom.author_chain_id,
                    residue_number=int(atom.author_seq_num),
                    insertion_code=insertion_code,
                )
                seq_idx = int(atom.mmcif_seq_num) - seq_start_num[atom.mmcif_chain_id]
                current = seq_to_structure_mappings.get(atom.author_chain_id, {})
                current[seq_idx] = ResidueAtPosition(
                    position=position,
                    name=atom.residue_name,
                    is_missing=False,
                    hetflag=hetflag,
                )
                seq_to_structure_mappings[atom.author_chain_id] = current

        # Add missing residue information to seq_to_structure_mappings.
        for chain_id, seq_info in valid_chains.items():
            author_chain = mmcif_to_author_chain_id[chain_id]
            current_mapping = seq_to_structure_mappings[author_chain]
            for idx, monomer in enumerate(seq_info):
                if idx not in current_mapping:
                    current_mapping[idx] = ResidueAtPosition(
                        position=None,
                        name=monomer.id,
                        is_missing=True,
                        hetflag=" ",
                    )

        author_chain_to_sequence = {}
        for chain_id, seq_info in valid_chains.items():
            author_chain = mmcif_to_author_chain_id[chain_id]
            seq = []
            for monomer in seq_info:
                code = PDBData.protein_letters_3to1_extended.get(monomer.id, "X")
                seq.append(code if len(code) == 1 else "X")
            seq = "".join(seq)
            author_chain_to_sequence[author_chain] = seq

        mmcif_object = MmcifObject(
            file_id=file_id,
            header=header,
            structure=first_model_structure,
            chain_to_seqres=author_chain_to_sequence,
            seqres_to_structure=seq_to_structure_mappings,
            raw_string=parsed_info,
        )

        return ParsingResult(mmcif_object=mmcif_object, errors=errors)
    except Exception as e:  # pylint:disable=broad-except
        errors[(file_id, "")] = e
        if not catch_all_errors:
            raise
        return ParsingResult(mmcif_object=None, errors=errors)


def _get_first_model(structure: PdbStructure) -> PdbStructure:
    """Returns the first model in a Biopython structure."""
    return next(structure.get_models())


_MIN_LENGTH_OF_CHAIN_TO_BE_COUNTED_AS_PEPTIDE = 21


def get_release_date(parsed_info: MmCIFDict) -> str:
    """Returns the oldest revision date."""
    revision_dates = parsed_info["_pdbx_audit_revision_history.revision_date"]
    return min(revision_dates)


def _get_header(parsed_info: MmCIFDict) -> PdbHeader:
    """Returns a basic header containing method, release date and resolution."""
    header = {}

    experiments = mmcif_loop_to_list("_exptl.", parsed_info)
    header["structure_method"] = ",".join(
        [experiment["_exptl.method"].lower() for experiment in experiments]
    )

    # Note: The release_date here corresponds to the oldest revision. We prefer to
    # use this for dataset filtering over the deposition_date.
    if "_pdbx_audit_revision_history.revision_date" in parsed_info:
        header["release_date"] = get_release_date(parsed_info)
    else:
        logging.warning(
            "Could not determine release_date: %s", parsed_info["_entry.id"]
        )

    header["resolution"] = 0.00
    for res_key in (
        "_refine.ls_d_res_high",
        "_em_3d_reconstruction.resolution",
        "_reflns.d_resolution_high",
    ):
        if res_key in parsed_info:
            try:
                raw_resolution = parsed_info[res_key][0]
                header["resolution"] = float(raw_resolution)
            except ValueError:
                pass
                # logging.info("Invalid resolution format: %s", parsed_info[res_key])

    return header


def _get_atom_site_list(parsed_info: MmCIFDict) -> Sequence[AtomSite]:
    """Returns list of atom sites; contains data not present in the structure."""
    return [
        AtomSite(*site)
        for site in zip(  # pylint:disable=g-complex-comprehension
            parsed_info["_atom_site.label_comp_id"],
            parsed_info["_atom_site.auth_asym_id"],
            parsed_info["_atom_site.label_asym_id"],
            parsed_info["_atom_site.auth_seq_id"],
            parsed_info["_atom_site.label_seq_id"],
            parsed_info["_atom_site.pdbx_PDB_ins_code"],
            parsed_info["_atom_site.group_PDB"],
            parsed_info["_atom_site.pdbx_PDB_model_num"],
        )
    ]


def _get_protein_chains(
    *, parsed_info: Mapping[str, Any]
) -> Mapping[ChainId, Sequence[Monomer]]:
    """Extracts polymer information for protein chains only.

    Args:
      parsed_info: _mmcif_dict produced by the Biopython parser.

    Returns:
      A dict mapping mmcif chain id to a list of Monomers.
    """
    # Get polymer information for each entity in the structure.
    entity_poly_seqs = mmcif_loop_to_list("_entity_poly_seq.", parsed_info)

    polymers = collections.defaultdict(list)
    for entity_poly_seq in entity_poly_seqs:
        polymers[entity_poly_seq["_entity_poly_seq.entity_id"]].append(
            Monomer(
                id=entity_poly_seq["_entity_poly_seq.mon_id"],
                num=int(entity_poly_seq["_entity_poly_seq.num"]),
            )
        )

    # Get chemical compositions. Will allow us to identify which of these polymers
    # are proteins.
    chem_comps = mmcif_loop_to_dict("_chem_comp.", "_chem_comp.id", parsed_info)

    # Get chains information for each entity. Necessary so that we can return a
    # dict keyed on chain id rather than entity.
    struct_asyms = mmcif_loop_to_list("_struct_asym.", parsed_info)

    entity_to_mmcif_chains = collections.defaultdict(list)
    for struct_asym in struct_asyms:
        chain_id = struct_asym["_struct_asym.id"]
        entity_id = struct_asym["_struct_asym.entity_id"]
        entity_to_mmcif_chains[entity_id].append(chain_id)

    # Identify and return the valid protein chains.
    valid_chains = {}
    for entity_id, seq_info in polymers.items():
        chain_ids = entity_to_mmcif_chains[entity_id]

        # Reject polymers without any peptide-like components, such as DNA/RNA.
        if any(
            [
                "peptide" in chem_comps[monomer.id]["_chem_comp.type"]
                for monomer in seq_info
            ]
        ):
            for chain_id in chain_ids:
                valid_chains[chain_id] = seq_info
    return valid_chains


def _is_set(data: str) -> bool:
    """Returns False if data is a special mmCIF character indicating 'unset'."""
    return data not in (".", "?")


# Modified to also get bfactor
def get_atom_coords_b(
    mmcif_object: MmcifObject, chain_id: str, _zero_center_positions: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Locate the right chain
    chains = list(mmcif_object.structure.get_chains())
    relevant_chains = [c for c in chains if c.id == chain_id]
    if len(relevant_chains) != 1:
        raise ValueError(f"Expected exactly one chain in structure with id {chain_id}.")
    chain = relevant_chains[0]

    # Extract the coordinates
    num_res = len(mmcif_object.chain_to_seqres[chain_id])
    all_atom_positions = np.zeros(
        [num_res, residue_constants.atom_type_num, 3], dtype=np.float32
    )
    all_atom_mask = np.zeros(
        [num_res, residue_constants.atom_type_num], dtype=np.float32
    )
    all_atom_b_factor = np.zeros(
        [num_res, residue_constants.atom_type_num], dtype=np.float32
    )
    for res_index in range(num_res):
        pos = np.zeros([residue_constants.atom_type_num, 3], dtype=np.float32)
        mask = np.zeros([residue_constants.atom_type_num], dtype=np.float32)
        res_at_position = mmcif_object.seqres_to_structure[chain_id][res_index]
        b_factors = np.zeros([residue_constants.atom_type_num], dtype=np.float32)
        if not res_at_position.is_missing:
            res = chain[
                (
                    res_at_position.hetflag,
                    res_at_position.position.residue_number,
                    res_at_position.position.insertion_code,
                )
            ]
            for atom in res.get_atoms():
                atom_name = atom.get_name()
                x, y, z = atom.get_coord()
                b_factor = atom.get_bfactor()
                if atom_name in residue_constants.atom_order.keys():
                    pos[residue_constants.atom_order[atom_name]] = [x, y, z]
                    mask[residue_constants.atom_order[atom_name]] = 1.0
                    b_factors[residue_constants.atom_order[atom_name]] = b_factor
                elif atom_name.upper() == "SE" and res.get_resname() == "MSE":
                    # Put the coords of the selenium atom in the sulphur column
                    pos[residue_constants.atom_order["SD"]] = [x, y, z]
                    mask[residue_constants.atom_order["SD"]] = 1.0
                    b_factors[residue_constants.atom_order["SD"]] = b_factor

        all_atom_positions[res_index] = pos
        all_atom_mask[res_index] = mask
        all_atom_b_factor[res_index] = b_factors

    if _zero_center_positions:
        binary_mask = all_atom_mask.astype(bool)
        translation_vec = all_atom_positions[binary_mask].mean(axis=0)
        all_atom_positions[binary_mask] -= translation_vec

    return all_atom_positions, all_atom_mask, all_atom_b_factor


FeatureDict = Mapping[str, np.ndarray]
ModelOutput = Mapping[str, Any]  # Is a nested dict.
PICO_TO_ANGSTROM = 0.01

PDB_CHAIN_IDS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
PDB_MAX_CHAINS = len(PDB_CHAIN_IDS)
assert PDB_MAX_CHAINS == 62


@dataclasses.dataclass(frozen=True)
class Protein:
    """Protein structure representation."""

    # Cartesian coordinates of atoms in angstroms. The atom types correspond to
    # residue_constants.atom_types, i.e. the first three are N, CA, CB.
    atom_positions: np.ndarray  # [num_res, num_atom_type, 3]

    # Amino-acid type for each residue represented as an integer between 0 and
    # 20, where 20 is 'X'.
    aatype: np.ndarray  # [num_res]

    # Binary float mask to indicate presence of a particular atom. 1.0 if an atom
    # is present and 0.0 if not. This should be used for loss masking.
    atom_mask: np.ndarray  # [num_res, num_atom_type]

    # Residue index as used in PDB. It is not necessarily continuous or 0-indexed.
    residue_index: np.ndarray  # [num_res]

    # B-factors, or temperature factors, of each residue (in sq. angstroms units),
    # representing the displacement of the residue from its ground truth mean
    # value.
    b_factors: np.ndarray  # [num_res, num_atom_type]

    # Chain indices for multi-chain predictions
    chain_index: Optional[np.ndarray] = None

    # Optional remark about the protein. Included as a comment in output PDB
    # files
    remark: Optional[str] = None

    # Templates used to generate this protein (prediction-only)
    parents: Optional[Sequence[str]] = None

    # Chain corresponding to each parent
    parents_chain_index: Optional[Sequence[int]] = None

    # Amino-acid sequence as a string, as read from file
    sequence: Optional[str] = None  # [num_res]

    def __post_init__(self):
        if len(np.unique(self.chain_index)) > PDB_MAX_CHAINS:
            raise ValueError(
                f"Cannot build an instance with more than {PDB_MAX_CHAINS} "
                "chains because these cannot be written to PDB format"
            )


def from_pdb_string(pdb_str: str, chain_id: Optional[str] = None) -> Protein:
    """Takes a PDB string and constructs a Protein object.

    WARNING: All non-standard residue types will be converted into UNK. All
      non-standard atoms will be ignored.

    Args:
      pdb_str: The contents of the pdb file
      chain_id: If None, then the whole pdb file is parsed. If chain_id is specified (e.g. A), then only that chain
        is parsed.

    Returns:
      A new `Protein` parsed from the pdb contents.
    """
    pdb_fh = io.StringIO(pdb_str)
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("none", pdb_fh)
    models = list(structure.get_models())
    if len(models) != 1:
        raise ValueError(
            f"Only single model PDBs are supported. Found {len(models)} models."
        )
    model = models[0]

    atom_positions = []
    sequence = []
    aatype = []
    atom_mask = []
    residue_index = []
    chain_ids = []
    b_factors = []

    for chain in model:
        if chain_id is not None and chain.id != chain_id:
            continue

        for res in chain:
            if res.id[2] != " ":
                raise ValueError(
                    f"PDB contains an insertion code at chain {chain.id} and residue "
                    f"index {res.id[1]}. These are not supported."
                )
            res_shortname = residue_constants.restype_3to1.get(res.resname, "X")
            restype_idx = residue_constants.restype_order.get(
                res_shortname, residue_constants.restype_num
            )
            pos = np.zeros((residue_constants.atom_type_num, 3))
            mask = np.zeros((residue_constants.atom_type_num,))
            res_b_factors = np.zeros((residue_constants.atom_type_num,))
            for atom in res:
                if atom.name not in residue_constants.atom_types:
                    continue
                pos[residue_constants.atom_order[atom.name]] = atom.coord
                mask[residue_constants.atom_order[atom.name]] = 1.0
                res_b_factors[residue_constants.atom_order[atom.name]] = atom.bfactor
            if np.sum(mask) < 0.5:
                # If no known atom positions are reported for the residue then skip it.
                continue
            
            sequence.append(res_shortname)
            aatype.append(restype_idx)
            atom_positions.append(pos)
            atom_mask.append(mask)
            residue_index.append(res.id[1])
            chain_ids.append(chain.id)
            b_factors.append(res_b_factors)

    parents = None
    parents_chain_index = None
    if "PARENT" in pdb_str:
        parents = []
        parents_chain_index = []
        chain_id = 0
        for l in pdb_str.split("\n"):
            if "PARENT" in l:
                if not "N/A" in l:
                    parent_names = l.split()[1:]
                    parents.extend(parent_names)
                    parents_chain_index.extend([chain_id for _ in parent_names])
                chain_id += 1

    unique_chain_ids = np.unique(chain_ids)
    if (len(unique_chain_ids) == 1) and (unique_chain_ids[0] == " "):
        chain_ids = ["A" for _ in chain_ids]

    chain_id_mapping = {cid: n for n, cid in enumerate(string.ascii_uppercase)}
    chain_index = np.array([chain_id_mapping[cid] for cid in chain_ids])

    return Protein(
        sequence="".join(sequence),
        atom_positions=np.array(atom_positions),
        atom_mask=np.array(atom_mask),
        aatype=np.array(aatype),
        residue_index=np.array(residue_index),
        chain_index=chain_index,
        b_factors=np.array(b_factors),
        parents=parents,
        parents_chain_index=parents_chain_index,
    )


def from_proteinnet_string(proteinnet_str: str) -> Protein:
    tag_re = r"(\[[A-Z]+\]\n)"
    tags = [tag.strip() for tag in re.split(tag_re, proteinnet_str) if len(tag) > 0]
    groups = zip(tags[0::2], [l.split("\n") for l in tags[1::2]])

    atoms = ["N", "CA", "C"]
    aatype = None
    atom_positions = None
    atom_mask = None
    seq = None
    for g in groups:
        if "[PRIMARY]" == g[0]:
            seq = g[1][0].strip()
            for i in range(len(seq)):
                if seq[i] not in residue_constants.restypes:
                    seq[i] = "X"
            aatype = np.array(
                [
                    residue_constants.restype_order.get(
                        res_symbol, residue_constants.restype_num
                    )
                    for res_symbol in seq
                ]
            )
        elif "[TERTIARY]" == g[0]:
            tertiary = []
            for axis in range(3):
                tertiary.append(list(map(float, g[1][axis].split())))
            tertiary_np = np.array(tertiary)
            atom_positions = np.zeros(
                (len(tertiary[0]) // 3, residue_constants.atom_type_num, 3)
            ).astype(np.float32)
            for i, atom in enumerate(atoms):
                atom_positions[:, residue_constants.atom_order[atom], :] = np.transpose(
                    tertiary_np[:, i::3]
                )
            atom_positions *= PICO_TO_ANGSTROM
        elif "[MASK]" == g[0]:
            mask = np.array(list(map({"-": 0, "+": 1}.get, g[1][0].strip())))
            atom_mask = np.zeros(
                (
                    len(mask),
                    residue_constants.atom_type_num,
                )
            ).astype(np.float32)
            for i, atom in enumerate(atoms):
                atom_mask[:, residue_constants.atom_order[atom]] = 1
            atom_mask *= mask[..., None]

    return Protein(
        atom_positions=atom_positions,
        atom_mask=atom_mask,
        aatype=aatype,
        sequence=seq,
        residue_index=np.arange(len(aatype)),
        b_factors=None,
    )


def _chain_end(atom_index, end_resname, chain_name, residue_index) -> str:
    chain_end = "TER"
    return (
        f"{chain_end:<6}{atom_index:>5}      {end_resname:>3} "
        f"{chain_name:>1}{residue_index:>4}"
    )


def get_pdb_headers(prot: Protein, chain_id: int = 0) -> Sequence[str]:
    pdb_headers = []

    remark = prot.remark
    if remark is not None:
        pdb_headers.append(f"REMARK {remark}")

    parents = prot.parents
    parents_chain_index = prot.parents_chain_index
    if parents_chain_index is not None:
        parents = [p for i, p in zip(parents_chain_index, parents) if i == chain_id]

    if parents is None or len(parents) == 0:
        parents = ["N/A"]

    pdb_headers.append(f"PARENT {' '.join(parents)}")

    return pdb_headers


def add_pdb_headers(prot: Protein, pdb_str: str) -> str:
    """Add pdb headers to an existing PDB string. Useful during multi-chain
    recycling
    """
    out_pdb_lines = []
    lines = pdb_str.split("\n")

    remark = prot.remark
    if remark is not None:
        out_pdb_lines.append(f"REMARK {remark}")

    parents_per_chain = None
    if prot.parents is not None and len(prot.parents) > 0:
        parents_per_chain = []
        if prot.parents_chain_index is not None:
            cur_chain = prot.parents_chain_index[0]
            parent_dict = {}
            for p, i in zip(prot.parents, prot.parents_chain_index):
                parent_dict.setdefault(str(i), [])
                parent_dict[str(i)].append(p)

            max_idx = max([int(chain_idx) for chain_idx in parent_dict])
            for i in range(max_idx + 1):
                chain_parents = parent_dict.get(str(i), ["N/A"])
                parents_per_chain.append(chain_parents)
        else:
            parents_per_chain.append(prot.parents)
    else:
        parents_per_chain = [["N/A"]]

    make_parent_line = lambda p: f"PARENT {' '.join(p)}"

    out_pdb_lines.append(make_parent_line(parents_per_chain[0]))

    chain_counter = 0
    for i, l in enumerate(lines):
        if "PARENT" not in l and "REMARK" not in l:
            out_pdb_lines.append(l)
        if "TER" in l and not "END" in lines[i + 1]:
            chain_counter += 1
            if not chain_counter >= len(parents_per_chain):
                chain_parents = parents_per_chain[chain_counter]
            else:
                chain_parents = ["N/A"]

            out_pdb_lines.append(make_parent_line(chain_parents))

    return "\n".join(out_pdb_lines)


def to_pdb(prot: Protein) -> str:
    """Converts a `Protein` instance to a PDB string.

    Args:
      prot: The protein to convert to PDB.

    Returns:
      PDB string.
    """
    restypes = residue_constants.restypes + ["X"]
    res_1to3 = lambda r: residue_constants.restype_1to3.get(restypes[r], "UNK")
    atom_types = residue_constants.atom_types

    pdb_lines = []

    atom_mask = prot.atom_mask
    aatype = prot.aatype
    atom_positions = prot.atom_positions
    residue_index = prot.residue_index.astype(np.int32)
    b_factors = prot.b_factors
    chain_index = prot.chain_index.astype(np.int32)

    if np.any(aatype > residue_constants.restype_num):
        raise ValueError("Invalid aatypes.")

    # Construct a mapping from chain integer indices to chain ID strings.
    chain_ids = {}
    for i in np.unique(chain_index):  # np.unique gives sorted output.
        if i >= PDB_MAX_CHAINS:
            raise ValueError(
                f"The PDB format supports at most {PDB_MAX_CHAINS} chains."
            )
        chain_ids[i] = PDB_CHAIN_IDS[i]

    headers = get_pdb_headers(prot)
    if len(headers) > 0:
        pdb_lines.extend(headers)

    pdb_lines.append("MODEL     1")
    n = aatype.shape[0]
    atom_index = 1
    last_chain_index = chain_index[0]
    prev_chain_index = 0
    chain_tags = string.ascii_uppercase

    # Add all atom sites.
    for i in range(aatype.shape[0]):
        # Close the previous chain if in a multichain PDB.
        if last_chain_index != chain_index[i]:
            pdb_lines.append(
                _chain_end(
                    atom_index,
                    res_1to3(aatype[i - 1]),
                    chain_ids[chain_index[i - 1]],
                    residue_index[i - 1],
                )
            )
            last_chain_index = chain_index[i]
            atom_index += 1  # Atom index increases at the TER symbol.

        res_name_3 = res_1to3(aatype[i])
        for atom_name, pos, mask, b_factor in zip(
            atom_types, atom_positions[i], atom_mask[i], b_factors[i]
        ):
            if mask < 0.5:
                continue

            record_type = "ATOM"
            name = atom_name if len(atom_name) == 4 else f" {atom_name}"
            alt_loc = ""
            insertion_code = ""
            occupancy = 1.00
            element = atom_name[0]  # Protein supports only C, N, O, S, this works.
            charge = ""

            chain_tag = "A"
            if chain_index is not None:
                chain_tag = chain_tags[chain_index[i]]

            # PDB is a columnar format, every space matters here!
            atom_line = (
                f"{record_type:<6}{atom_index:>5} {name:<4}{alt_loc:>1}"
                # TODO: check this refactor, chose main branch version
                # f"{res_name_3:>3} {chain_ids[chain_index[i]]:>1}"
                f"{res_name_3:>3} {chain_tag:>1}"
                f"{residue_index[i]:>4}{insertion_code:>1}   "
                f"{pos[0]:>8.3f}{pos[1]:>8.3f}{pos[2]:>8.3f}"
                f"{occupancy:>6.2f}{b_factor:>6.2f}          "
                f"{element:>2}{charge:>2}"
            )
            pdb_lines.append(atom_line)
            atom_index += 1

        should_terminate = i == n - 1
        if chain_index is not None:
            if i != n - 1 and chain_index[i + 1] != prev_chain_index:
                should_terminate = True
                prev_chain_index = chain_index[i + 1]

        if should_terminate:
            # Close the chain.
            chain_end = "TER"
            chain_termination_line = (
                f"{chain_end:<6}{atom_index:>5}      "
                f"{res_1to3(aatype[i]):>3} "
                f"{chain_tag:>1}{residue_index[i]:>4}"
            )
            pdb_lines.append(chain_termination_line)
            atom_index += 1

            if i != n - 1:
                # "prev" is a misnomer here. This happens at the beginning of
                # each new chain.
                pdb_lines.extend(get_pdb_headers(prot, prev_chain_index))

    pdb_lines.append("ENDMDL")
    pdb_lines.append("END")

    # Pad all lines to 80 characters
    pdb_lines = [line.ljust(80) for line in pdb_lines]
    return "\n".join(pdb_lines) + "\n"  # Add terminating newline.


def to_modelcif(prot: Protein) -> str:
    """
    Converts a `Protein` instance to a ModelCIF string. Chains with identical modelled coordinates
    will be treated as the same polymer entity. But note that if chains differ in modelled regions,
    no attempt is made at identifying them as a single polymer entity.

    Args:
      prot: The protein to convert to PDB.

    Returns:
      ModelCIF string.
    """

    restypes = residue_constants.restypes + ["X"]
    atom_types = residue_constants.atom_types

    atom_mask = prot.atom_mask
    aatype = prot.aatype
    atom_positions = prot.atom_positions
    residue_index = prot.residue_index.astype(np.int32)
    b_factors = prot.b_factors
    chain_index = prot.chain_index

    n = aatype.shape[0]
    if chain_index is None:
        chain_index = [0 for i in range(n)]

    system = modelcif.System(title="OpenFold prediction")

    # Finding chains and creating entities
    seqs = {}
    seq = []
    last_chain_idx = None
    for i in range(n):
        if last_chain_idx is not None and last_chain_idx != chain_index[i]:
            seqs[last_chain_idx] = seq
            seq = []
        seq.append(restypes[aatype[i]])
        last_chain_idx = chain_index[i]
    # finally add the last chain
    seqs[last_chain_idx] = seq

    # now reduce sequences to unique ones (note this won't work if different asyms have different unmodelled regions)
    unique_seqs = {}
    for chain_idx, seq_list in seqs.items():
        seq = "".join(seq_list)
        if seq in unique_seqs:
            unique_seqs[seq].append(chain_idx)
        else:
            unique_seqs[seq] = [chain_idx]

    # adding 1 entity per unique sequence
    entities_map = {}
    for key, value in unique_seqs.items():
        model_e = modelcif.Entity(key, description="Model subunit")
        for chain_idx in value:
            entities_map[chain_idx] = model_e

    chain_tags = string.ascii_uppercase
    asym_unit_map = {}
    for chain_idx in set(chain_index):
        # Define the model assembly
        chain_id = chain_tags[chain_idx]
        asym = modelcif.AsymUnit(
            entities_map[chain_idx], details="Model subunit %s" % chain_id, id=chain_id
        )
        asym_unit_map[chain_idx] = asym
    modeled_assembly = modelcif.Assembly(
        asym_unit_map.values(), name="Modeled assembly"
    )

    class _LocalPLDDT(modelcif.qa_metric.Local, modelcif.qa_metric.PLDDT):
        name = "pLDDT"
        software = None
        description = "Predicted lddt"

    class _GlobalPLDDT(modelcif.qa_metric.Global, modelcif.qa_metric.PLDDT):
        name = "pLDDT"
        software = None
        description = "Global pLDDT, mean of per-residue pLDDTs"

    class _MyModel(modelcif.model.AbInitioModel):
        def get_atoms(self):
            # Add all atom sites.
            for i in range(n):
                for atom_name, pos, mask, b_factor in zip(
                    atom_types, atom_positions[i], atom_mask[i], b_factors[i]
                ):
                    if mask < 0.5:
                        continue
                    element = atom_name[
                        0
                    ]  # Protein supports only C, N, O, S, this works.
                    yield modelcif.model.Atom(
                        asym_unit=asym_unit_map[chain_index[i]],
                        type_symbol=element,
                        seq_id=residue_index[i],
                        atom_id=atom_name,
                        x=pos[0],
                        y=pos[1],
                        z=pos[2],
                        het=False,
                        biso=b_factor,
                        occupancy=1.00,
                    )

        def add_scores(self):
            # local scores
            plddt_per_residue = {}
            for i in range(n):
                for mask, b_factor in zip(atom_mask[i], b_factors[i]):
                    if mask < 0.5:
                        continue
                    # add 1 per residue, not 1 per atom
                    if chain_index[i] not in plddt_per_residue:
                        # first time a chain index is seen: add the key and start the residue dict
                        plddt_per_residue[chain_index[i]] = {residue_index[i]: b_factor}
                    if residue_index[i] not in plddt_per_residue[chain_index[i]]:
                        plddt_per_residue[chain_index[i]][residue_index[i]] = b_factor
            plddts = []
            for chain_idx in plddt_per_residue:
                for residue_idx in plddt_per_residue[chain_idx]:
                    plddt = plddt_per_residue[chain_idx][residue_idx]
                    plddts.append(plddt)
                    self.qa_metrics.append(
                        _LocalPLDDT(
                            asym_unit_map[chain_idx].residue(residue_idx), plddt
                        )
                    )
            # global score
            self.qa_metrics.append((_GlobalPLDDT(np.mean(plddts))))

    # Add the model and modeling protocol to the file and write them out:
    model = _MyModel(assembly=modeled_assembly, name="Best scoring model")
    model.add_scores()

    model_group = modelcif.model.ModelGroup([model], name="All models")
    system.model_groups.append(model_group)

    fh = io.StringIO()
    modelcif.dumper.write(fh, [system])
    return fh.getvalue()


def ideal_atom_mask(prot: Protein) -> np.ndarray:
    """Computes an ideal atom mask.

    `Protein.atom_mask` typically is defined according to the atoms that are
    reported in the PDB. This function computes a mask according to heavy atoms
    that should be present in the given sequence of amino acids.

    Args:
      prot: `Protein` whose fields are `numpy.ndarray` objects.

    Returns:
      An ideal atom mask.
    """
    return residue_constants.STANDARD_ATOM_MASK[prot.aatype]


def from_prediction(
    features: FeatureDict,
    result: ModelOutput,
    b_factors: Optional[np.ndarray] = None,
    remove_leading_feature_dimension: bool = True,
    remark: Optional[str] = None,
    parents: Optional[Sequence[str]] = None,
    parents_chain_index: Optional[Sequence[int]] = None,
) -> Protein:
    """Assembles a protein from a prediction.

    Args:
      features: Dictionary holding model inputs.
      result: Dictionary holding model outputs.
      b_factors: (Optional) B-factors to use for the protein.
      remove_leading_feature_dimension: Whether to remove the leading dimension
        of the `features` values
      chain_index: (Optional) Chain indices for multi-chain predictions
      remark: (Optional) Remark about the prediction
      parents: (Optional) List of template names
    Returns:
      A protein instance.
    """

    def _maybe_remove_leading_dim(arr: np.ndarray) -> np.ndarray:
        return arr[0] if remove_leading_feature_dimension else arr

    if "asym_id" in features:
        chain_index = _maybe_remove_leading_dim(features["asym_id"]) - 1
    else:
        chain_index = np.zeros_like(_maybe_remove_leading_dim(features["aatype"]))

    if b_factors is None:
        b_factors = np.zeros_like(result["final_atom_mask"])

    aatype = _maybe_remove_leading_dim(features["aatype"])
    sequence = "".join([residue_constants.restypes[i.item()] for i in aatype])

    return Protein(
        aatype=aatype,
        sequence=sequence,
        atom_positions=result["final_atom_positions"],
        atom_mask=result["final_atom_mask"],
        residue_index=_maybe_remove_leading_dim(features["residue_index"]) + 1,
        b_factors=b_factors,
        chain_index=chain_index,
        remark=remark,
        parents=parents,
        parents_chain_index=parents_chain_index,
    )
