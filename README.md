# Can Watermarks Survive Translation? On the Cross-lingual Consistency of Text Watermark for Large Language Models

* This repository contains the implementation of X-SIR in *[&#34;Can Watermarks Survive Translation? On the Cross-lingual Consistency of Text Watermark for Large Language Models&#34;](https://arxiv.org/abs/2402.14007)*.
* It is adapted and mostly unchanged from [SIR](https://github.com/THU-BPM/Robust_Watermark).



### Conda Environment

Tested on the following environment, but it should work on other versions.

- python 3.10.10
- pytorch
- `pip3 install -r requirements.txt`



### Step1: Generate embeddings for training watermark model

```shell
python3 generate_embeddings.py \
	--input_path data/sts/train.jsonl \
	--output_path data/embeddings/train_embeddings.txt \
	--model_path paraphrase-multilingual-mpnet-base-v2 \
	--size 2000
```



### Step2: Train watermark model

* Train the watermark model using the embeddings generated in Step1

  ```shell
  python3 train_watermark_model.py \
  	--input_path data/embeddings/train_embeddings.txt \
  	--output_model model/transform_model_x-sbert.pth \
  	--input_dim 768
  ```

* [Optional] You could check the quality of the trained model by running the following command to visualize the similarity:

  ```shell
  python3 analysis_transform_model.py \
  	--embedding_file data/embeddings/train_embeddings.txt \
  	--input_dim 768 \
  	--checkpoint model/transform_model_x-sbert.pth \
  	--figure_dir data/figures/
  ```

  It should be like:

  <p align="center">
  <img src="assert/imgs/origin_graph.png" alt="origin_graph"  width="500" />
  </p>



### Step3: Generate watermarked & translated text 

- Generate mapping files

  - SIR

    ```shell
    python3 generate_mappings.py \
      --model baichuan-inc/Baichuan-7B \
      --output_file data/mappings/300_mapping_baichuan.json
    ```

  - X-SIR

    ```diff
    python3 generate_semantic_mappings.py \
      --model baichuan-inc/Baichuan-7B \
    + --dictionary data/mappings/en-zh_dict.txt \
    - --output_file data/mappings/300_mapping_baichuan.json
    + --output_file data/mappings/300_mapping_baichuan_enzh.json
    ```
    
    *You can check the semantic clustering of tokens in `data/mappings/300_mapping_baichuan_enzh_cluster.json`.*
    
    

- Generate watermarked text (use `text summarization` as an example: )

  - SIR

    ```shell
    python3 inference_with_watermark.py \
      --base_model baichuan-inc/Baichuan-7B \
      --prmopt_file data/dataset/multinews_cwra_prompt.json \
      --output_file gen/baichuan/multinews_sir_output.json \
      --mapping_file data/mappings/300_mapping_baichuan.json \
      --transform_model model/transform_model_x-sbert.pth \
      --embedding_model paraphrase-multilingual-mpnet-base-v2
    ```

  * X-SIR

    ```diff
    python3 inference_with_watermark.py \
      --base_model baichuan-inc/Baichuan-7B \
      --prmopt_file data/dataset/multinews_cwra_prompt.json \
    - --output_file gen/baichuan/multinews_sir_output.json \
    - --mapping_file data/mappings/300_mapping_baichuan.json \
    + --output_file gen/baichuan/multinews_x-sir_output.json \
    + --mapping_file data/mappings/300_mapping_baichuan_enzh.json \
      --transform_model model/transform_model_x-sbert.pth \
      --embedding_model paraphrase-multilingual-mpnet-base-v2
    ```



* Translate the watermarked text into English

  * SIR

    ```shell
    python3 zh2en_translate.py \
    	--input_file  gen/baichuan/multinews_sir_output.json \
    	--output_file gen/baichuan/multinews_sir_trans.json
    ```

  * X-SIR

    ```shell
    python3 zh2en_translate.py \
    	--input_file  gen/baichuan/multinews_x-sir_output.json \
    	--output_file gen/baichuan/multinews_x-sir_trans.json
    ```




### Step4: Compute watermark strength (z-score)

* SIR

  ```shell
  # reference
  python3 detect.py \
  	--base_model baichuan-inc/Baichuan-7B \
    --detect_file data/dataset/multinews_ref.json \
    --output_path gen/baichuan/multinews_sir_ref_zscore.json \
    --mapping_file data/mappings/300_mapping_baichuan.json \
    --transform_model model/transform_model_x-sbert.pth \
    --embedding_model paraphrase-multilingual-mpnet-base-v2
  
  # output
  python3 detect.py \
    --base_model baichuan-inc/Baichuan-7B \
    --detect_file gen/baichuan/multinews_sir_output.json \
    --output_path gen/baichuan/multinews_sir_output_zscore.json \
    --mapping_file data/mappings/300_mapping_baichuan.json \
    --transform_model model/transform_model_x-sbert.pth \
    --embedding_model paraphrase-multilingual-mpnet-base-v2
  
  # translated output
  python3 detect.py \
    --base_model baichuan-inc/Baichuan-7B \
    --detect_file gen/baichuan/multinews_sir_trans.json \
    --output_path gen/baichuan/multinews_sir_trans_zscore.json \
    --mapping_file data/mappings/300_mapping_baichuan.json \
    --transform_model model/transform_model_x-sbert.pth \
    --embedding_model paraphrase-multilingual-mpnet-base-v2
  ```

* X-SIR

  ```diff
  # reference
  python3 detect.py \
    --base_model baichuan-inc/Baichuan-7B \
    --detect_file data/dataset/multinews_ref.json \
  - --output_path gen/baichuan/multinews_sir_ref_zscore.json \
  - --mapping_file data/mappings/300_mapping_baichuan.json \
  + --output_path gen/baichuan/multinews_x-sir_ref_zscore.json \
  + --mapping_file data/mappings/300_mapping_baichuan_enzh.json \
    --transform_model model/transform_model_x-sbert.pth \
    --embedding_model paraphrase-multilingual-mpnet-base-v2
  
  # output
  python3 detect.py \
    --base_model baichuan-inc/Baichuan-7B \
  - --detect_file gen/baichuan/multinews_sir_output.json \
  - --output_path gen/baichuan/multinews_sir_output_zscore.json \
  - --mapping_file data/mappings/300_mapping_baichuan.json \
  + --detect_file gen/baichuan/multinews_x-sir_output.json \
  + --output_path gen/baichuan/multinews_x-sir_output_zscore.json \
  + --mapping_file data/mappings/300_mapping_baichuan_enzh.json \
    --transform_model model/transform_model_x-sbert.pth \
    --embedding_model paraphrase-multilingual-mpnet-base-v2
  
  # translated output
  python3 detect.py \
    --base_model baichuan-inc/Baichuan-7B \
  - --detect_file gen/baichuan/multinews_sir_trans.json \
  - --output_path gen/baichuan/multinews_sir_trans_zscore.json \
  - --mapping_file data/mappings/300_mapping_baichuan.json \
  + --detect_file gen/baichuan/multinews_x-sir_trans.json \
  + --output_path gen/baichuan/multinews_x-sir_trans_zscore.json \
  + --mapping_file data/mappings/300_mapping_baichuan_enzh.json \
    --transform_model model/transform_model_x-sbert.pth \
    --embedding_model paraphrase-multilingual-mpnet-base-v2
  ```



### Step5: Compute AUC

* SIR

  ```shell
  # No attack
  python3 auc.py \
  	--ref_zscore gen/baichuan/multinews_sir_ref_zscore.json \
  	--wm_zscore gen/baichuan/multinews_sir_output_zscore.json
  # AUC: 0.9327999999999999
  
  # CWRA
  python3 auc.py \
  	--ref_zscore gen/baichuan/multinews_sir_ref_zscore.json \
  	--wm_zscore gen/baichuan/multinews_sir_trans_zscore.json
  # AUC: 0.64686
  ```

  

* X-SIR

  ```shell
  # No attack
  python3 auc.py \
  	--ref_zscore gen/baichuan/multinews_x-sir_ref_zscore.json \
  	--wm_zscore gen/baichuan/multinews_x-sir_output_zscore.json
  # AUC: 0.9388799999999999
  
  # CWRA
  python3 auc.py \
  	--ref_zscore gen/baichuan/multinews_x-sir_ref_zscore.json \
  	--wm_zscore gen/baichuan/multinews_x-sir_trans_zscore.json
  # AUC: 0.80324
  ```




## Citation

```ruby
@article{he2024can,
  title={Can Watermarks Survive Translation? On the Cross-lingual Consistency of Text Watermark for Large Language Models},
  author={He, Zhiwei and Zhou, Binglin and Hao, Hongkun and Liu, Aiwei and Wang, Xing and Tu, Zhaopeng and Zhang, Zhuosheng and Wang, Rui},
  journal={arXiv preprint arXiv:2402.14007},
  year={2024}
}
```
