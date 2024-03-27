# Running on the digs first

qlogin -c 2 --mem=4g -p gpu --gres=gpu:gtx1050:1 # for production rtx2080

qlogin -c 4 --mem=16g -p gpu --gres=gpu:rtx2080:1 

-c 2 --mem=4g 
#
/home/rdkibler/software/alphafold/superfold --mock_msa_depth 512 --models 1 2 3 4 5 --type monomer_ptm --version monomer  --max_recycles 6 --out_dir OUT_DIR  OUT_DIR/P1_P2.fasta OUT_DIR/P1_P3.fasta 
