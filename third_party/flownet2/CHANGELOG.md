Main changes to the repository:
Change imread, imsave to be from imageio instead of scipy.misc
Changed similar lines like the following 
`PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)`
to
`PYBIND11_MODULE(channelnorm_cuda, m)`.

Please see commit history for more detailed change logs.
