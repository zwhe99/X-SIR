### Implementing X-SIR from Scratch

X-SIR requires two addtional file:

* A **watermark/transform model** that receives an embedding and produces biases for the next-token logits. In fact, a watermark model yields a vector with size (300) far smaller than the vocab size (>32K).
* A **mapping file** that maps the token id to its logit bias.



#### Watermark model

##### Generate embeddings for training watermark model

```shell
python3 src_watermark/xsir/generate_embeddings.py \
	--input_path data/dataset/sts/train.jsonl \
	--output_path data/embedding/train_embeddings_10K.txt \
	--model_path paraphrase-multilingual-mpnet-base-v2 \
	--size 10000
```

##### Train watermark model

* Train the watermark model using the embeddings

  ```shell
  python src_watermark/xsir/train_watermark_model.py \
      --input_path data/embedding/train_embeddings_10K.txt \
      --output_model data/model/transform_model_x-sbert_10K.pth \
      --input_dim 768
  ```

* [Optional] You could check the quality of the trained model by running the following command to visualize the similarity:

  ```shell
  python3 src_watermark/xsir/analysis_transform_model.py \
  	--embedding_file data/embedding/train_embeddings_10K.txt \
  	--input_dim 768 \
  	--checkpoint data/model/transform_model_x-sbert_10K.pth \
  	--figure_dir data/figures
  ```

  It should be like:

  <p align="center">
  <img src="assert/origin_graph.png" alt="origin_graph"  width="400" />
  </p>


#### Mapping file

##### Download external dictionaries

```shell
wget -P data/dictionary/download https://dl.fbaipublicfiles.com/arrival/dictionaries/de-en.txt
wget -P data/dictionary/download https://dl.fbaipublicfiles.com/arrival/dictionaries/en-de.txt
wget -P data/dictionary/download https://dl.fbaipublicfiles.com/arrival/dictionaries/de-fr.txt
wget -P data/dictionary/download https://dl.fbaipublicfiles.com/arrival/dictionaries/fr-de.txt
wget -P data/dictionary/download https://dl.fbaipublicfiles.com/arrival/dictionaries/fr-en.txt
wget -P data/dictionary/download https://dl.fbaipublicfiles.com/arrival/dictionaries/en-fr.txt
wget -P data/dictionary/download https://dl.fbaipublicfiles.com/arrival/dictionaries/zh-en.txt
wget -P data/dictionary/download https://dl.fbaipublicfiles.com/arrival/dictionaries/en-zh.txt
wget -P data/dictionary/download https://dl.fbaipublicfiles.com/arrival/dictionaries/ja-en.txt
wget -P data/dictionary/download https://dl.fbaipublicfiles.com/arrival/dictionaries/en-ja.txt
```

##### Build a unidentified dictionary

```shell
python3 data/dictionary/build_dictionary.sh \
    --dicts \
        data/dictionary/download/de-en.txt \
        data/dictionary/download/de-fr.txt \
        data/dictionary/download/en-de.txt \
        data/dictionary/download/en-fr.txt \
        data/dictionary/download/en-ja.txt \
        data/dictionary/download/en-zh.txt \
        data/dictionary/download/fr-de.txt \
        data/dictionary/download/fr-en.txt \
        data/dictionary/download/ja-en.txt \
        data/dictionary/download/zh-en.txt \
    --output_file data/dictionary/dictionary.txt \
    --add_meta_symbols
```

##### Build a mapping file for your model (semantic clustering in the paper)

```shell
MODEL_NAME=baichuan-inc/Baichuan-7B
MODEL_ABBR=baichuan-7b
python3 src_watermark/xsir/generate_semantic_mappings.py \
    --model $MODEL_NAME \
    --dictionary data/dictionary/dictionary.txt \
    --output_file data/mapping/xsir/300_mapping_$MODEL_ABBR.json
```