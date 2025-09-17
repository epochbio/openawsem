# ----------------------------------------------------------------------
# Copyright (2010) Aram Davtyan and Garegin Papoian
#
# Papoian's Group, University of Maryland at Collage Park
# http://papoian.chem.umd.edu/
#
# Last Update: 07/18/2011
# -------------------------------------------------------------------------

#import sys

#if len(sys.argv)!=5:
#	print
#	print " > " + sys.argv[0] + " fasta.file pdb_file index_file chain_id"
#	print
#	exit()

from Bio.Align import PairwiseAligner
from Bio.PDB.PDBParser import PDBParser
from Bio import SeqIO
import os
from typing import List, Tuple


def three2one(prot: list
              ) -> str:
    """
    Translate a protein sequence from 3-letter to 1-letter code.

    Args:
        prot: A list representing the protein sequence in 3-letter code.

    Returns:
        A string representing the protein sequence in 1-letter code.
    """

    code = {"GLY" : "G", "ALA" : "A", "LEU" : "L", "ILE" : "I",
            "ARG" : "R", "LYS" : "K", "MET" : "M", "CYS" : "C",
            "TYR" : "Y", "THR" : "T", "PRO" : "P", "SER" : "S",
            "TRP" : "W", "ASP" : "D", "GLU" : "E", "ASN" : "N",
            "GLN" : "Q", "PHE" : "F", "HIS" : "H", "VAL" : "V",
            "M3L" : "K", "MSE" : "M", "CAS" : "C" }

    newprot = ""
    for a in prot:
        newprot += code.get(a, "X")

    return newprot


def getListOfValidAlignments(alignments: list, 
                             pdb_indexes: list
                             ) -> list:
    """
    Get a list of indices for valid alignments.

    Args:
        alignments: A list of tuples, where each tuple contains two aligned sequences.
        pdb_indexes: A list of integers representing the PDB indexes of the residues.

    Returns:
        A list of integers representing the indices of the valid alignments in the input list.
    """
    ialigns = []
    for i in range(0, len(alignments)):
        alignment = alignments[i]
        alseq_fasta = alignment[0]
        alseq_pdb = alignment[1]

        if len(alseq_fasta)!=len(alseq_pdb):
            print("Error using alignment too")
            print()
            exit()

        last_i_pdb = -1
        last_j = -1
        i_pdb = 0
        valid = True
        for j in range(0, len(alseq_pdb)):
                if alseq_pdb[j]!='-':
    #                                if last_i_pdb!=-1 and j - last_j > i_pdb - last_i_pdb:
                        if last_i_pdb!=-1 and j - last_j != pdb_indexes[i_pdb] - pdb_indexes[last_i_pdb]:
                                valid = False
                                break
                        last_i_pdb = i_pdb
                        last_j = j
                        i_pdb = i_pdb + 1
        if valid:
                ialigns.append(i)
    return ialigns


def getFastaSequence(fasta_file: str
                     ) -> str:
    """Reads a FASTA file and returns the sequence as a string.

    Args:
        fasta_file (str): The path to the FASTA file.

    Returns:
        str: The sequence extracted from the FASTA file.
    """
    inFASTA=open(fasta_file, 'r')
    inseq=SeqIO.read(inFASTA,'fasta')

    return str(inseq.seq)


def getPdbSequance(pdb_file: str, 
                   chain_id: str
                   ) -> Tuple[str, List[int]]:
    """
    Extracts the sequence and residue numbers from a PDB file for a specific chain.

    Args:
        pdb_file: The path to the PDB file.
        chain_id: The identifier of the chain to extract the sequence from.

    Returns:
        A tuple containing the sequence of residues in one-letter code and a list of residue numbers.
    """
    pdb_indexes = []
    pdb_sequance = []

    p = PDBParser(PERMISSIVE=1, QUIET=True)
    s = p.get_structure("", pdb_file)
    pdb_id = pdb_file[0:-4]

    if not s[0].has_id(chain_id):
        print("PDB "+pdb_id+" doesn't have chain with id "+chain_id)
        print()
        exit()

    chain = s[0][chain_id]

    ires = 0
    for res in chain:
        is_regular_res = res.has_id('N') and res.has_id('CA') and res.has_id('C') and (res.get_resname()=='GLY' or res.has_id('CB'))
        res_id = res.get_id()[0]
        if (res_id ==' ' or res_id =='H_MSE' or res_id =='H_M3L' or res_id =='H_CAS') and is_regular_res:
                ires = ires + 1
                res_name = res.get_resname()
                residue_no = res.get_id()[1]
                pdb_sequance.append(res_name)
                pdb_indexes.append(residue_no)
        elif res_id !='W':
                print("Unknown residue in "+pdb_id+" with res_id "+res_id)

    pdb_seq = three2one(pdb_sequance)

    return pdb_seq, pdb_indexes


def getIndexArray(alignment: tuple, 
                  pdb_indexes: list
                  ) -> list:
    """Generates an array of indices mapping the alignment to PDB indexes.

    Args:
        alignment: A tuple containing two aligned sequences (FASTA sequence and PDB sequence).
        pdb_indexes: A list of integers representing the PDB indexes of the residues.

    Returns:
        A list of lists, where each inner list contains the index in the FASTA sequence,
        the corresponding index in the PDB sequence (or -1 if not aligned), and the residue.
    """
    alseq_fasta = alignment[0]
    alseq_pdb = alignment[1]

    index_array = []
    i_fasta = 0
    i_pdb = 0
    for i in range(0, len(alseq_fasta)):
        if alseq_fasta[i]!='-':
            index_pdb = -1
            if alseq_pdb[i]!='-': index_pdb = pdb_indexes[i_pdb]
            index_array.append([i_fasta, index_pdb, alseq_fasta[i]])
            i_fasta = i_fasta + 1
        if alseq_pdb[i]!='-':
            i_pdb = i_pdb + 1

    return index_array


def writeIndexFile(fasta_file: str, 
                   pdb_file: str, 
                   index_file: str, 
                   chain_id: str):
    """Writes an index file mapping the residues of a FASTA sequence to a PDB file.

    Args:
        fasta_file: The path to the FASTA file containing the protein sequence.
        pdb_file: The path to the PDB file containing the protein structure.
        index_file: The path where the index file will be written.
        chain_id: The chain identifier in the PDB file to be indexed.
    """
#	from Bio import pairwise2
#	from Bio.PDB.PDBParser import PDBParser
#	from Bio import SeqIO

    pdb_id = pdb_file[0:-4]

#fasta_file = sys.argv[1]
#pdb_id = sys.argv[2]
#pdb_file = pdb_id+".pdb"
#index_file = sys.argv[3]
#chain_id = sys.argv[4]

    fasta_seq = ""
    pdb_seq = ""
    pdb_indexes = []
    answer = ""
    shift = 0
    index_list = []

    fasta_seq = getFastaSequence(fasta_file)
    pdb_seq, pdb_indexes = getPdbSequance(pdb_file, chain_id)

    # Do a check on pdb_indexes, if some residue is in wrong order. for example in PDB 5cxv, line 3383
    # ATOM   2762  OH  TYR A1160     -21.292 -22.385  85.747  1.00 48.40           O
    # ATOM   2763  N   PHE A 355     -16.598 -16.950  80.885  1.00 51.66           N
    # The residue 1160 is ahead of residue 355.
    preIndex = pdb_indexes[0]
    needRenumber = False
    for i in pdb_indexes[1:]:
        if i < preIndex:
            print(preIndex, i, "need reorder")
            needRenumber = True
            break
        preIndex = i

    if needRenumber:
        pdbLocation = os.path.dirname(pdb_file)
        os.system(f"mv {pdb_file} {pdb_file}.bak")
        try:
            os.system(f"pdb_reres {pdb_file}.bak > {pdb_file}")
            pdb_seq, pdb_indexes = getPdbSequance(pdb_file, chain_id)
        except:
            print("You might want to install pdb-tools by doing something like 'pip install pdb-tools'")

    print()
    print(fasta_seq)
    print(len(fasta_seq))
    print(pdb_seq)
    print(pdb_indexes)
    print(len(pdb_seq))

    print()
    print()

    if len(fasta_seq)==len(pdb_seq) and pdb_indexes[0]==1 and pdb_indexes[-1]==len(pdb_seq):
        print("FULLMATCH")
        answer = "FULLMATCH"
    elif len(fasta_seq)==len(pdb_seq) and pdb_indexes[-1]-pdb_indexes[0]+1==len(pdb_seq):
        print("Indexes are simply shifted by " + str(pdb_indexes[0]-1))
        answer = "SHIFT"
        shift = pdb_indexes[0]-1
    elif len(fasta_seq)==len(pdb_seq) and fasta_seq==pdb_seq:
        print("Number is messed up")
        print("Same length")
        answer = "INDEXED"
        for i in range(0, len(fasta_seq)):
            if i!=0 and pdb_indexes[i]<= pdb_indexes[i-1]:
                answer = "SKIP"
                index_list = []
                break
            index_list.append([ i, pdb_indexes[i], fasta_seq[i] ])
    else:
        aligner = PairwiseAligner()
        aligner.mode = 'global'  # for global alignment
        aligner.match_score = 2
        aligner.mismatch_score = -1
        aligner.open_gap_score = -0.5
        aligner.extend_gap_score = -0.1
        alignments = aligner.align(fasta_seq, pdb_seq)
        #print alignments
        #print len(alignments)
        #print

        alist = getListOfValidAlignments(alignments, pdb_indexes)
        #print len(alist), alist

        if len(alist)==1:
            answer = "INDEXED"
            index_list = getIndexArray(alignments[alist[0]], pdb_indexes)
        elif len(alist)>1:
            answer = "SKIP"
        elif len(alist)==0:
            answer = "SKIP"
            #alignments = pairwise2.align.globalxx(fasta_seq, pdb_seq)
            #print
            #print alignments
                #print len(alignments)
            #print

            #alist2 = getListOfValidAlignments(alignments, pdb_indexes)
                #print len(alist2), alist2

            #if len(alist2)==1:
            #	answer = "INDEXED"
            #	index_list = getIndexArray(alignments[alist2[0]], pdb_indexes)
            #elif len(alist2)>1:
            #	answer = "SKIP"
            #elif len(alist2)==0:
            #	answer = "SKIP"

    out = open(index_file, 'w')
    out.write(answer)
    if answer=="SHIFT":
        out.write("\n")
        out.write(str(shift))
    elif answer=="INDEXED":
        for ind in index_list:
            out.write("\n")
            out.write(str(ind[0]+1))
            out.write(" ")
            out.write(str(ind[1]))
            out.write(" ")
            out.write(ind[2])
    out.close()


#fasta_file = sys.argv[1]
#pdb_id = sys.argv[2]
#pdb_file = pdb_id+".pdb"
#index_file = sys.argv[3]
#chain_id = sys.argv[4]

#writeIndexFile(fasta_file, pdb_file, index_file, chain_id)
