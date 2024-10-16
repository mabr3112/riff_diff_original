#!/home/mabr3112/anaconda3/bin/python3.9
#
#
#
###########################################

def mutate_seq(fasta_file: str, pos: int, aa: str) -> None:
    '''
    '''
    pos += -1
    with open(fasta_file, 'r') as f:
        c = [x.strip() for x in f.readlines() if x]
    
    c[1] = c[1][:pos] + aa + c[1][pos+1:]
    with open(fasta_file, 'w') as f:
        f.write("\n".join(c))

    return None
