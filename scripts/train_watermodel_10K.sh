set -e # Exit immediately if a command exits with a non-zero status.
set -u # Treat unset variables as an error when substituting.

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
WORK_DIR=$SCRIPT_DIR/..
DATA_DIR=$WORK_DIR/data
GEN_DIR=$WORK_DIR/gen
ATTACK_DIR=$WORK_DIR/attack

python3 $WORK_DIR/src_watermark/xsir/generate_embeddings.py \
	--input_path $DATA_DIR/dataset/sts/train.jsonl \
	--output_path $DATA_DIR/embedding/train_embeddings_10K.txt \
	--model_path paraphrase-multilingual-mpnet-base-v2 \
	--size 10000

python $WORK_DIR/src_watermark/xsir/train_watermark_model.py \
    --input_path $DATA_DIR/embedding/train_embeddings_10K.txt \
    --output_model $DATA_DIR/model/transform_model_x-sbert_10K.pth \
    --input_dim 768

python3 $WORK_DIR/src_watermark/xsir/analysis_transform_model.py \
	--embedding_file $DATA_DIR/embedding/train_embeddings_10K.txt \
	--input_dim 768 \
	--checkpoint $DATA_DIR/model/transform_model_x-sbert_10K.pth \
	--figure_dir $DATA_DIR/figures/