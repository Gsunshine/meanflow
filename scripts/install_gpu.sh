conda install -c nvidia cudnn=9.8

pip install pillow clu tensorflow==2.15.0 "keras<3" "torch<=2.4" torchvision tensorflow_datasets matplotlib==3.9.2
pip install orbax-checkpoint==0.4.4 ml-dtypes==0.5.0 tensorstore==0.1.67
pip install diffusers dm-tree cached_property

# test with jax and jaxlib 0.6.2, replace `jax.tree_leaves` with `jax.tree.leaves` in code
pip install -U "jax[cuda12]" 
