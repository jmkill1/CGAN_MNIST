# CGAN_MNIST
```
27训练CGAN
cgan-mnist1.py 
	line91 model_name = 'cgan27' #保存的模型名前缀
	line138 y_sample[:, 2] = 1
	line118 saver.restore(sess, 'checkpoints/27/cgan27')
	line134 save(saver, sess,'./checkpoints/27')
	
python cgan-mnist1.py --epoch 50000 --nlog 1000 --data ./data/27/mnist_data_27 --outpath ./out/27/T_50000
python cgan-mnist1.py --epoch 20000 --nlog 400 --data ./data/27/mnist_data_72 --outpath ./out/27/F_20000 --load True
python cgan-mnist1.py --epoch 8000 --nlog 160 --data ./data/27/mnist_data_27 --outpath ./out/27/T_8000 --load True
python cgan-mnist1.py --epoch 3200 --nlog 64 --data ./data/27/mnist_data_72 --outpath ./out/27/F_3200 --load True
python cgan-mnist1.py --epoch 1280 --nlog 25 --data ./data/27/mnist_data_27 --outpath ./out/27/T_1280 --load True
python cgan-mnist1.py --epoch 512 --nlog 10 --data ./data/27/mnist_data_72 --outpath ./out/27/F_512 --load True


46训练CGAN
cgan-mnist1.py 
	line91 model_name = 'cgan46' #保存的模型名前缀
	line138 y_sample[:, 4] = 1
	line118 saver.restore(sess, 'checkpoints/46/cgan46')
	line134 save(saver, sess,'./checkpoints/46')
python cgan-mnist1.py --epoch 50000 --nlog 1000 --data ./data/46/mnist_data_46 --outpath ./out/46/T_50000
python cgan-mnist1.py --epoch 20000 --nlog 400 --data ./data/46/mnist_data_64 --outpath ./out/46/F_20000 --load True
python cgan-mnist1.py --epoch 8000 --nlog 160 --data ./data/46/mnist_data_46 --outpath ./out/46/T_8000 --load True
python cgan-mnist1.py --epoch 3200 --nlog 64 --data ./data/46/mnist_data_64 --outpath ./out/46/F_3200 --load True
python cgan-mnist1.py --epoch 1280 --nlog 25 --data ./data/46/mnist_data_46 --outpath ./out/46/T_1280 --load True
python cgan-mnist1.py --epoch 512 --nlog 10 --data ./data/46/mnist_data_64 --outpath ./out/46/F_512 --load True

49训练CGAN
cgan-mnist1.py 
	line91 model_name = 'cgan49' #保存的模型名前缀
	line138 y_sample[:, 4] = 1
	line118 saver.restore(sess, 'checkpoints/49/cgan49')
	line134 save(saver, sess,'./checkpoints/49')
python cgan-mnist1.py --epoch 50000 --nlog 1000 --data ./data/49/mnist_data_49 --outpath ./out/49/T_50000
python cgan-mnist1.py --epoch 20000 --nlog 400 --data ./data/49/mnist_data_94 --outpath ./out/49/F_20000 --load True
python cgan-mnist1.py --epoch 8000 --nlog 160 --data ./data/49/mnist_data_49 --outpath ./out/49/T_8000 --load True
python cgan-mnist1.py --epoch 3200 --nlog 64 --data ./data/49/mnist_data_94 --outpath ./out/49/F_3200 --load True
python cgan-mnist1.py --epoch 1280 --nlog 25 --data ./data/49/mnist_data_49 --outpath ./out/49/T_1280 --load True
python cgan-mnist1.py --epoch 512 --nlog 10 --data ./data/49/mnist_data_94 --outpath ./out/49/F_512 --load True

68训练CGAN
cgan-mnist1.py 
	line91 model_name = 'cgan68' #保存的模型名前缀
	line138 y_sample[:, 6] = 1
	line118 saver.restore(sess, 'checkpoints/68/cgan68')
	line134 save(saver, sess,'./checkpoints/68')
python cgan-mnist1.py --epoch 50000 --nlog 1000 --data ./data/68/mnist_data_68 --outpath ./out/68/T_50000
python cgan-mnist1.py --epoch 20000 --nlog 400 --data ./data/68/mnist_data_86 --outpath ./out/68/F_20000 --load True
python cgan-mnist1.py --epoch 8000 --nlog 160 --data ./data/68/mnist_data_68 --outpath ./out/68/T_8000 --load True
python cgan-mnist1.py --epoch 3200 --nlog 64 --data ./data/68/mnist_data_86 --outpath ./out/68/F_3200 --load True
python cgan-mnist1.py --epoch 1280 --nlog 25 --data ./data/68/mnist_data_68 --outpath ./out/68/T_1280 --load True
python cgan-mnist1.py --epoch 512 --nlog 10 --data ./data/68/mnist_data_86 --outpath ./out/68/F_512 --load True

生成27、46、49、68触发集
create_tirgger.py line94 y_sample[:, 2] = 1 改成你要生成的图片one-hot编码
python create_trigger.py --ckpt checkpoints/27/cgan27 --outpath ./out/27/trigger
create_tirgger.py line94 y_sample[:, 4] = 1 改成你要生成的图片one-hot编码
python create_trigger.py --ckpt checkpoints/46/cgan46 --outpath ./out/46/trigger
python create_trigger.py --ckpt checkpoints/49/cgan49 --outpath ./out/49/trigger
create_tirgger.py line94 y_sample[:, 6] = 1 改成你要生成的图片one-hot编码
python create_trigger.py --ckpt checkpoints/68/cgan68 --outpath ./out/68/trigger

model/mnist_cnn.h5	未嵌入水印的深度学习模型
chaos.py		混搭标注
	line37 将路径改为要混沌标记图片文件夹的路径
	line39 保存路径
	27 x=3.999 u=0.88 interval=0.5
	46 x=3.999 u=0.88 interval=0.5
	49 x=3.989 u=0.87 interval=0.4
	68 x=3.989 u=0.87 interval=0.4

将触发集和训练集混合到一起形成新的数据集来训练模型嵌入水印的能力
Mnist.h5 --val_accuracy 0.9943 
Mnist64.h5  --val_accuracy 0.9934 --extraction 1
Mnist128.h5  --val_accuracy 0.9946 --extraction 1
Mnist256.h5 --val_accuracy 0.9934 --extraction 1

微调
	val_acc 	0.9926	32o 0.9688 	32u 0.875
	val_acc 	0.9929	64o 0.96 	64u 1.0
	val_acc	0.9932	128o 0.99 	128u 1.0

覆盖
	64		-val_accuracy 0.9934	extraction 1.0 1.0		size 12.52MB
		0.1	-val_accuracy 0.9928 	extraction 1.0 1.0		size 10.88MB
		0.2	-val_accuracy 0.9922	extraction 1.0 1.0		size 9.98MB
		0.3	-val_accuracy 0.9921	extraction 0.9375 1.0		size 9.08MB
		0.4	-val_accuracy 0.9924	extraction 0.9375 0.96875		size 8.10MB
		0.5	-val_accuracy 0.9918	extraction 0.90625 0.90625	size 7.08MB
	128		-val_accuracy 0.9946	extraction 1.0 1.0		size 12.52MB
		0.1	-val_accuracy 0.9940 	extraction 1.0 1.0		size 10.90MB
		0.2	-val_accuracy 0.9937	extraction 1.0 1.0		size 10.03MB
		0.3	-val_accuracy 0.9937	extraction 1.0 1.0		size 9.12MB
		0.4	-val_accuracy 0.9937	extraction 1.0 1.0		size 8.14MB
		0.5	-val_accuracy 0.9934	extraction 1.0 0.96875		size 7.11MB
	256		-val_accuracy 0.9934	extraction 1.0 1.0		size 12.52MB
		0.1	-val_accuracy 0.9933	extraction 1.0 1.0		size 10.93MB
		0.2	-val_accuracy 0.9930	extraction 1.0 1.0		size 10.10MB
		0.3	-val_accuracy 0.9930	extraction 1.0 1.0		size 9.16MB
		0.4	-val_accuracy 0.9929	extraction 1.0 1.0		size 8.14MB
		0.5	-val_accuracy 0.9917	extraction 0.9844 0.968751		size 7.13MB

Overwriting
	chaos.py		混搭标注
	line37 将路径改为要混沌标记图片文件夹的路径
	line39 保存路径
	27 x=3.900 u=0.91 interval=0.45
	46 x=3.900 u=0.91 interval=0.45
	49 x=3.900 u=0.91 interval=0.45
	68 x=3.900 u=0.91 interval=0.45
	128 
		owner_extraction 1.0 user_extraction 0.9843 wm 0.5781 acc 0.9940
		owner_extraction 0.626 user_extraction 0.625 wm 0.7578 acc 0.8828
	64	
		owner_extraction 0.75 user_extraction 1.0 wm 0.5781 acc 0.9918
		owner_extraction 0.46875 user_extraction 0.5937 wm 0.6718 acc 0.8916
	256
		owner_extraction 1.0 user_extraction 1.0 wm 0.5936 acc 0.9934
		owner_extraction 0.6640 user_extraction 0.6484 wm 0.6445 acc 0.9017
		
	128		-val_accuracy 0.9934	extraction 1.0 1.0		size 12.52MB
		0.1	-val_accuracy 0.9928 	extraction 1.0 1.0		size 10.88
		0.2	-val_accuracy 0.9922	extraction 1.0 1.0		size 9.98MB
		0.3	-val_accuracy 0.9921	extraction 0.9375 1.0		size 9.08MB
		0.4	-val_accuracy 0.9924	extraction 0.9375 0.96875		size 8.10MB
		0.4	-val_accuracy 0.9918	extraction 0.90625 0.90625	size 7.08MB
```
