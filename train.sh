#added
export MKL_THREADING_LAYER=GNU

output="runs/basic_fullcanon"
# rm ${output}/parser.log
device="cpu"
dataset="django.pnet.fullcanon.dataset.freq3.par_info.refact.space_only.bin"

# django dataset
echo "training django dataset"
commandline="-batch_size 10 -max_epoch 50 -valid_per_batch 4000 -save_per_batch 4000 -decode_max_time_step 100 -optimizer adam -rule_embed_dim 128 -node_embed_dim 64 -valid_metric bleu -concat_type basic -include_cid True"
# -model ./runs/fixed/model.iter16000.npz"
datatype="django"

# train the model
THEANO_FLAGS="mode=FAST_RUN,device=${device},floatX=float32,traceback.limit=20" python -u code_gen.py \
	-data_type ${datatype} \
	-data data/${dataset} \
	-output_dir ${output} \
	${commandline} \
	train
echo "Done training!!"

# decode testing set, and evaluate the model which achieves the best bleu and accuracy, resp.
for model in "model.best_bleu.npz" "model.best_acc.npz"; do
	THEANO_FLAGS="mode=FAST_RUN,device=${device},floatX=float32,exception_verbosity=high" python code_gen.py \
	-data_type ${datatype} \
	-data data/${dataset} \
	-output_dir ${output} \
	-model ${output}/${model} \
	${commandline} \
	decode \
	-saveto ${output}/${model}.decode_results.test.bin

	python code_gen.py \
		-data_type ${datatype} \
		-data data/${dataset} \
		-output_dir ${output} \
		evaluate \
		-input ${output}/${model}.decode_results.test.bin
done
