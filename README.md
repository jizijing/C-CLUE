# Baselines  
(1) BERT: It is a pre-training model proposed by Google AI Research Institute in October 2018. It is a milestone model achievement in the history of NLP development.  
(2) BERT-wwm: wwm is Whole Word Masking (Mask the whole word). Compared to Bert, its improvement is to replace a complete word with a Mask tag instead of a subword.  
(3) RoBERTa: The RoBERTa model uses larger model parameters, larger bacth size, and more training data. Besides, in the training method, Roberta removes the next sentence prediction task, adopts dynamic masking, and uses text encoding.  
(4) Zhongkeda-BERT: It uses the Daizhige corpus and the Tang Poetry and Song Ci data sets for further pre-training, and modified the maximum sentence length from 128 to 512. In addition, a restricted beam search is set up to exclude illegal transfers.  

# Implementation Details   
We follow the pre-training hyper-parameters used in BERT. For fine-tuning, most hyper-parameters are the same as pre-training, except batch size, learning rate, and number of training epochs. We find the following ranges of possible values work well on the training datasets with gold annotations, i.e., batch size: 32, learning rate (Adam): 5e-5, 3e-5, 2e-5, number of epochs ranging from 3 to 10.   

# File Discription  
data_ner 文件夹：包含NER任务的训练、校验以及测试数据集。  
new_classification_data_guwen 文件夹：包含关系分类任务的训练、校验以及测试数据集。  
sequence_labeling_data_guwen 文件夹：包含关系数据序列标注任务的训练、校验以及测试数据集。  
raw_data_guwen 文件夹：古文关系标注原始数据。  
run_NER.py:命名实体识别微调代码。  
predicate_data_manager.py:将原始数据处理为关系分类任务数据集的代码。  
run_predicate_classification.py：关系分类任务微调代码。  
sequence_labeling_data_manager.py：将原始数据处理为关系数据序列标注任务数据集的代码。  
run_sequnce_labeling.py：关系数据序列标注任务微调代码。  
prepare_data_for_labeling_infer.py：把关系分类模型预测结果转换成序列标注模型的预测输入。  
produce_submit_json_file.py：生成关系抽取结果。  
evaluate_classification.py:关系分类性能评测代码。  
evaluate_labeling.py:关系抽取性能评测代码。  

# 命名实体识别运行命令  
```
python run_NER.py \
          --task_name=NER \
          --do_train=true \
          --do_eval=true \
          --do_predict=true \
          --data_dir=../tmp/brandnew\  #数据文件夹路径
          --vocab_file=chinese_L-12_H-768_A-12/vocab.txt \ #模型词汇表
          --bert_config_file=chinese_L-12_H-768_A-12/bert_config.json \ #模型配置文件
          --learning_rate=2e-5 \
          --train_batch_size=32 \
          --num_train_epochs=3 \
          --output_dir=output/epochs3 \ #输出文件夹路径
          --init_checkpoint=chinese_L-12_H-768_A-12/bert_model.ckpt\  #模型文件
          --max_seq_length=256 \     # 根据实际句子长度可调
```
# 关系抽取过程

**关系分类模型和实体序列标注模型可以同时训练，但是只能依次预测！**  

## 训练阶段  

### 准备关系分类数据  
```
python predicate_data_manager.py
```

### 关系分类模型训练  
```
python run_predicate_classification.py \
--task_name=SKE_2019 \
--do_train=true \
--do_eval=false \
--data_dir=new_classification_data_guwen \
--vocab_file=chinese_L-12_H-768_A-12/vocab.txt \
--bert_config_file=chinese_L-12_H-768_A-12/bert_config.json \
--init_checkpoint=chinese_L-12_H-768_A-12/bert_model.ckpt \
--max_seq_length=128 \
--train_batch_size=32 \
--learning_rate=2e-5 \
--num_train_epochs=3 \
--output_dir=output/predicate_classification_model/epochs3/
```

### 准备序列标注数据  
```
python sequence_labeling_data_manager.py
```

### 序列标注模型训练  
```
python run_sequnce_labeling.py \
--task_name=SKE_2019 \
--do_train=true \
--do_eval=false \
--data_dir=sequence_labeling_data_guwen \
--vocab_file=chinese_L-12_H-768_A-12/vocab.txt \
--bert_config_file=chinese_L-12_H-768_A-12/bert_config.json \
--init_checkpoint=chinese_L-12_H-768_A-12/bert_model.ckpt \
--max_seq_length=128 \
--train_batch_size=32 \
--learning_rate=2e-5 \
--num_train_epochs=3 \
--output_dir=output/sequnce_labeling_model/epochs3/
```

## 预测阶段

### 关系分类模型预测
```
python run_predicate_classification.py \
  --task_name=SKE_2019 \
  --do_predict=true \
  --data_dir=new_classification_data_guwen \
  --vocab_file=chinese_L-12_H-768_A-12/vocab.txt \
  --bert_config_file=chinese_L-12_H-768_A-12/bert_config.json \
  --init_checkpoint=output/predicate_classification_model/epochs3/model.ckpt \
  --max_seq_length=128 \
  --output_dir=output/predicate_infer_out/epochs3
 ```

### 把关系分类模型预测结果转换成序列标注模型的预测输入
```
python prepare_data_for_labeling_infer.py
```

### 序列标注模型预测
```
python run_sequnce_labeling.py \
  --task_name=SKE_2019 \
  --do_predict=true \
  --data_dir=sequence_labeling_data_guwen \
  --vocab_file=chinese_L-12_H-768_A-12/vocab.txt \
  --bert_config_file=chinese_L-12_H-768_A-12/bert_config.json \
  --init_checkpoint=output/sequnce_labeling_model/epochs3/model.ckpt \
  --max_seq_length=128 \
  --output_dir=output/sequnce_infer_out/epochs3
 ```

### 生成实体-关系结果
```
python produce_submit_json_file.py
```

## 评估阶段
```
python evaluate_classification.py
python evaluate_labeling.py
```
