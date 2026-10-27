# Data and external artifacts

Obtain TOMATO-Star raw train/test and TOMATO-Star-SFT-Data-R1D-32B IR conversations from their authors under their separate terms. See model identifiers in configs. Place raw files at data/raw/tomato_star/{train,test}.jsonl and data/raw/sft/IR/train.jsonl. Use separate training/evaluation tree directories. Raw fields include source_id, research_question, background_survey and inspiration (found_title/supposed_title, found_abstract). Title matching is lowercase whitespace-normalized exact matching.

No source records, historical outcomes, scores, weights or generated result figures/tables are included. Exact historical tree/checkpoint URLs have not been supplied. configs/historical_tree_identity.json contains only the three evaluation-tree file hashes; hashes do not provide downloads or verify weights. Rebuilt trees are new realizations. Obtain or construct your own artifacts following reproduction.md.

Concept PNG panels are manuscript review assets with redistribution terms awaiting author confirmation. The code license does not establish figure, data or model terms. No remote resources were downloaded for this revision.
