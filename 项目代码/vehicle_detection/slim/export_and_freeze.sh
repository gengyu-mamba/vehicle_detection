TEMP_GRAPH_FILE=car/inception_v3_inf_graph.pb
DATASET_DIR=car/train_dir/
DATASET_NAME=car
OUTPUT_GRAPH=car/freezen_graph.pb
python export_inference_graph.py \
  --model_name=inception_v3 \
  --output_file=$TEMP_GRAPH_FILE \
  --dataset_dir=$DATASET_DIR \
  --dataset_name=$DATASET_NAME

python freeze_graph.py \
  --input_graph=$TEMP_GRAPH_FILE \
  --input_checkpoint=car/train_dir/model.ckpt-118847 \
  --output_node_names=output,final_probs \
  --input_binary=True \
  --output_graph=$OUTPUT_GRAPH
  
cp ${DATASET_DIR%*/}/labels.txt ${OUTPUT_GRAPH%*.pb}.label
