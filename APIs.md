# C APIs
## Free functions:
### NewDevice(device_type: 'int' = 0) -> 'int':
```
Create a new OIDN device

    Args:
        device_type(int) : OIDN_DEVICE_TYPE_XXX
```
### CommitDevice(device_handle: 'int'):
```
Batch up multiple changes on the device.

    Args:
        device_handle : Get from NewDevice
```
### GetDeviceError(device_handle: 'int') -> 'tuple[int, str]':
```
Args:
        device_handle : Get from NewDevice
    Returns:
        tuple of (error_code : int, error_message : str)
```
### ReleaseDevice(device_handle: 'int'):
```
Args:
        device_handle : Get from NewDevice
```
### RetainDevice(device_handle: 'int'):
```
Args:
        device_handle : Get from NewDevice
```
### SetDeviceBool(device_handle: 'int', name: 'str', value: 'bool'):
```
These parameters can be set by SetDeviceBool:
        setAffinity (default=True): enables thread affinitization (pinning software threads to hardware threads) if it is necessary for achieving optimal performance
    Args:
        device_handle : Get from NewDevice
        name : parameter name
        value : parameter value(bool type)
```
### SetDevice1b(device_handle: 'int', name: 'str', value: 'bool'):
```
Alias for SetDeviceBool
```
### SetDeviceInt(device_handle: 'int', name: 'str', value: 'int'):
```
These parameters can be set by SetDevice1b:
        verbose : 0 verbosity level of the console output between 0-4; when set to 0, no output is printed, when set to a higher level more output is printed
        numThreads (default = 0) : maximum number of threads which the library should use; 0 will set it automatically to get the best performance
    Args:
        device_handle : Get from NewDevice
        name : parameter name
        value : parameter value(bool type)
```
### SetDevice1i(device_handle: 'int', name: 'str', value: 'int'):
```
Alias for SetDeviceInt
```
### GetDeviceInt(device_handle: 'int', name: 'str') -> 'int':
```
These parameters can be get by GetDeviceInt:
        version : combined version number (major.minor.patch) with two decimal digits per component
        versionMajor : major version number
        versionMinor : minor version number
        versionPatch : patch version number
        verbose : 0 verbosity level of the console output between 0-4; when set to 0, no output is printed, when set to a higher level more output is printed
    Args:
        device_handle : Get from NewDevice
        name : parameter name
```
### GetDevice1i(device_handle: 'int', name: 'str') -> 'int':
```
Alias for GetDeviceInt
```
### GetDeviceBool(device_handle: 'int', name: 'str') -> 'bool':
```
These parameters can be get by GetDeviceBool:
        setAffinity (default = True): enables thread affinitization (pinning software threads to hardware threads) if it is necessary for achieving optimal performance
    Args:
        device_handle : Get from NewDevice
        name : parameter name
```
### GetDevice1b(device_handle: 'int', name: 'str') -> 'bool':
```
Alias for GetDeviceBool
```
### NewFilter(device_handle: 'int', type: 'str') -> 'int':
```
Creates a new filter of the specified type (e.g. "RT")

    Args:
        device_handle(int) : Created by NewDevice
        type(str) : e.g. "RT“ or "RTLightmap"
```
### SetSharedFilterImage(filter_handle: 'int', name: 'str', data: 'np.ndarray', format: 'int', width: 'int', height: 'int', byteOffset: 'int' = 0, bytePixelStride: 'int' = 0, byteRowStride: 'int' = 0):
```
Set filter image, the parameter name cound be:
        color : input beauty image (3 channels, LDR values in [0, 1] or HDR values in [0, +∞), values being interpreted such that, after scaling with the inputScale parameter, a value of 1 corresponds to a luminance level of 100 cd/m²)
        albedo(only support RT filter) : input auxiliary image containing the albedo per pixel (3 channels, values in [0, 1])
        normal(only support RT filter) : input auxiliary image containing the shading normal per pixel (3 channels, world-space or view-space vectors with arbitrary length, values in [-1, 1])
        output : output image (3 channels); can be one of the input images
    Args:
        filter_handle(int): Created by NewFilter
        name(str): color/albedo/normal/output, See document of OIDN.
        data(np.array): data buffer, should be correct in size and dtype; should be c_contiguous when name == 'output'
        format:  Should be oidn.FORMAT_FLOAT3 for image
        width: width in pixel.
        height: height in pixel.
        byteOffset: default to 0
        bytePixel: default to 0
        byteRawStride: default to 0
```
### UnsetFilterImage(filter_handle: 'int', name: 'str'):
```
Remove filter image, name could be color | albedo | normal | output
    Args:
        filter_handle : handle of fitler, get from NewFitler
        name : image name
```
### RemoveFilterImage(filter_handle: 'int', name: 'str'):
```
Alias for UnsetFilterImage
```
### SetSharedFilterData(filter_handle: 'int', name: 'str', data: 'np.array'):
```
Set filter data, the name could be:
        weights : trained model weights blob
    Args:
        filter_handle : Get from NewFilter
        name : name of the parameter
        data : numpy array with dtype != object
```
### UpdateFilterData(filter_handle: 'int', name: 'str'):
```
Just notify the filter that the contents of its data has been changed, name can be weight.

    Args:
        fitler_handle : Get from NewFilter
        name : parameter name
```
### UnsetFilterData(filter_handle: 'int', name: 'str'):
```
Remove the filter data, name can be weight.

    Args:
        fitler_handle : Get from NewFilter
        name : name of the data
```
### RemoveFilterData(filter_handle: 'int', name: 'str'):
```
Alias for UnsetFilterData
```
### GetFilterInt(filter_handle: 'int', name: 'str') -> 'int':
```
Get filter parameter (type int), the name could be:
        maxMemoryMB (default=3000) : approximate maximum scratch memory to use in megabytes (actual memory usage may be higher); limiting memory usage may cause slower denoising due to internally splitting the image into overlapping tiles
        alignment : when manually denoising in tiles, the tile size and offsets should be multiples of this amount of pixels to avoid artifacts; when denoising HDR images inputScale must be set by the user to avoid seam artifacts
        overlap : when manually denoising in tiles, the tiles should overlap by this amount of pixels
    Args:
        filter_handle : Get from NewFilter
        name : name of the parameter
```
### GetFilter1i(filter_handle: 'int', name: 'str') -> 'int':
```
Alias for GetFilterInt
```
### GetFilterBool(filter_handle: 'int', name: 'str') -> 'bool':
```
Get filter parameter (type bool), the name could be:
        hdr (default = False, only support RT) : whether the main input image is HDR
        srgb (default = False, only support RT) : whether the main input image is encoded with the sRGB (or 2.2 gamma) curve (LDR only) or is linear; the output will be encoded with the same curve
        cleanAux (default = False, only support RT) : whether the auxiliary feature (albedo, normal) images are noise-free; recommended for highest quality but should not be enabled for noisy auxiliary images to avoid residual noise
        directional (default = False, only support RTLightmap) : whether the input contains normalized coefficients (in [-1, 1]) of a directional lightmap (e.g. normalized L1 or higher spherical harmonics band with the L0 band divided out); if the range of the coefficients is different from [-1, 1], the inputScale parameter can be used to adjust the range without changing the stored values
    Args:
        filter_handle : Get from NewFilter
        name : name of the parameter
```
### GetFilter1b(filter_handle: 'int', name: 'str') -> 'bool':
```
Alias for GEtFilterBool
```
### GetFilterFloat(filter_handle: 'int', name: 'str') -> 'float':
```
Get filter parameter (type float), the name could be:
        inputScale (default=nan) : scales values in the main input image before filtering, without scaling the output too, which can be used to map color or auxiliary feature values to the expected range, e.g. for mapping HDR values to physical units (which affects the quality of the output but not the range of the output values); if set to NaN, the scale is computed implicitly for HDR images or set to 1 otherwise
    Args:
        filter_handle : Get from NewFilter
        name : name of the parameter
```
### GetFilter1f(filter_handle: 'int', name: 'str') -> 'float':
```
Alias for GetFilterFloat
```
### SetFilterBool(filter_handle: 'int', name: 'str', value: 'bool'):
```
Get filter parameter (type bool), the name could be:
        hdr (default = False, only support RT) : whether the main input image is HDR
        srgb (default = False, only support RT) : whether the main input image is encoded with the sRGB (or 2.2 gamma) curve (LDR only) or is linear; the output will be encoded with the same curve
        cleanAux (default = False, only support RT) : whether the auxiliary feature (albedo, normal) images are noise-free; recommended for highest quality but should not be enabled for noisy auxiliary images to avoid residual noise
        directional (default = False, only support RTLightmap) : whether the input contains normalized coefficients (in [-1, 1]) of a directional lightmap (e.g. normalized L1 or higher spherical harmonics band with the L0 band divided out); if the range of the coefficients is different from [-1, 1], the inputScale parameter can be used to adjust the range without changing the stored values
    Args:
        filter_handle : Get from NewFilter
        name : name of the parameter
        value : value of the parameter
```
### SetFilter1b(filter_handle: 'int', name: 'str', value: 'bool'):
```
Alias for SetFilterBool
```
### SetFilterInt(filter_handle: 'int', name: 'str', value: 'int'):
```
Get filter parameter (type int), the name could be:
        maxMemoryMB (default=3000) : approximate maximum scratch memory to use in megabytes (actual memory usage may be higher); limiting memory usage may cause slower denoising due to internally splitting the image into overlapping tiles
        alignment : when manually denoising in tiles, the tile size and offsets should be multiples of this amount of pixels to avoid artifacts; when denoising HDR images inputScale must be set by the user to avoid seam artifacts
        overlap : when manually denoising in tiles, the tiles should overlap by this amount of pixels
    Args:
        filter_handle : Get from NewFilter
        name : name of the parameter
        value : value of the parameter
```
### SetFilter1i(filter_handle: 'int', name: 'str', value: 'int'):
```
Alias for SetFilterInt
```
### SetFilterFloat(filter_handle: 'int', name: 'str', value: 'float'):
```
Get filter parameter (type float), the name could be:
        inputScale (default=nan) : scales values in the main input image before filtering, without scaling the output too, which can be used to map color or auxiliary feature values to the expected range, e.g. for mapping HDR values to physical units (which affects the quality of the output but not the range of the output values); if set to NaN, the scale is computed implicitly for HDR images or set to 1 otherwise
    Args:
        filter_handle : Get from NewFilter
        name : name of the parameter
        value : value of the parameter
```
### SetFilter1f(filter_handle: 'int', name: 'str', value: 'float'):
```
Alias for SetFilterFloat
```
### CommitFilter(filter_handle: 'int'):
```
Batch up multiple changes for the filter

    Args:
        filter_handle : Get from NewFilter
```
### ExecuteFilter(filter_handle: 'int'):
```
Execute the filter. Remember CommitFilter to ensure all your parameters notified.

    Args:
        filter_handle : Get from NewFilter
```
### ReleaseFilter(filter_handle: 'int'):
```
Release the filter

    Args:
        filter_handle : Get from NewFilter
```
### RetainFilter(filter_handle: 'int'):
```
Release the filter

    Args:
        filter_handle : Get from NewFilter
```
## Class RawFunction
These functions are FFI objects for corresponding native functions in the OIDN dynamic linked library.
### oidnCommitDevice
### oidnCommitFilter
### oidnExecuteFilter
### oidnGetDeviceBool
### oidnGetDeviceError
### oidnGetDeviceInt
### oidnGetFilterBool
### oidnGetFilterFloat
### oidnGetFilterInt
### oidnNewDevice
### oidnNewFilter
### oidnReleaseDevice
### oidnReleaseFilter
### oidnRetainDevice
### oidnRetainFilter
### oidnSetDeviceBool
### oidnSetDeviceInt
### oidnSetFilterBool
### oidnSetFilterFloat
### oidnSetFilterInt
### oidnSetSharedFilterData
### oidnSetSharedFilterImage
### oidnUnsetFilterData
### oidnUnsetFilterImage
### oidnUpdateFilterData
# Pythonic APIs
## Class Device
### \_\_init\_\_(self, backend: 'Backend | str | None' = None, *, device_type: 'Backend | str | None' = None, options: 'DeviceOptions | None' = None) -> 'None': <div style="text-align: right; float: right; color: #215f11">method</div> 
```
Create an OIDN device.

        Args:
            backend: backend selection (cpu/cuda/hip/sycl/metal/auto)
            device_type: legacy alias for backend selection
            options: device configuration options
```
### device\_handle <div style="text-align: right; float: right; color: #21138d">property</div>  
```
Returns the device handle
```
### error <div style="text-align: right; float: right; color: #21138d">property</div>  
```
Returns a tuple[error_code, error_message], the same as oidn.GetDeviceError.
```
### is\_cpu <div style="text-align: right; float: right; color: #21138d">property</div>  
```
Indicate whether it is a CPU device.
```
### is\_cuda <div style="text-align: right; float: right; color: #21138d">property</div>  
```
Indicate wheter it is a CUDA device.
```
### is\_hip <div style="text-align: right; float: right; color: #21138d">property</div>  
```
Indicate whether it is a HIP device.
```
### is\_metal <div style="text-align: right; float: right; color: #21138d">property</div>  
```
Indicate whether it is a Metal device.
```
### is\_sycl <div style="text-align: right; float: right; color: #21138d">property</div>  
```
Indicate whether it is a SYCL device.
```
### raise\_if\_error(self): <div style="text-align: right; float: right; color: #215f11">method</div> 
```
Raise a RuntimeError if an error occured.
```
### release(self): <div style="text-align: right; float: right; color: #215f11">method</div> 
```
Call ReleaseDevice with self.device_handle
```
## Class Filter
### \_\_init\_\_(self, device: 'Device', type: 'str') -> 'None': <div style="text-align: right; float: right; color: #215f11">method</div> 
```
Args:
            device : oidn.Device
            type   : 'RT' or 'RTLightmap'
```
### execute(self) -> 'None': <div style="text-align: right; float: right; color: #215f11">method</div> 
```
Run the filter, wait until finished.
```
### filter\_handle <div style="text-align: right; float: right; color: #21138d">property</div>  
```
Returns the handle of filter.
```
### release(self) -> 'None': <div style="text-align: right; float: right; color: #215f11">method</div> 
```
Call ReleaseFilter with self.filter_handle
```
### set\_image(self, name: 'str', buffer: 'Buffer') -> 'None': <div style="text-align: right; float: right; color: #215f11">method</div> 
```
Set image buffer for the filter.

        Args:
            name    : color/albedo/normal/output
            buffer  : Buffer object
```
### set\_images(self, *, color: 'Buffer', output: 'Buffer', albedo: 'Buffer | None' = None, normal: 'Buffer | None' = None) -> 'None': <div style="text-align: right; float: right; color: #215f11">method</div> 
```
Set the filter images in a single call.
```
## Class Buffer
### \_\_init\_\_(self, device: 'Device', buffer_delegate: 'object', format: 'int', channel_order: 'ChannelOrder', width: 'int', height: 'int', channels: 'int', byte_offset: 'int', byte_pixel_stride: 'int', byte_row_stride: 'int', data_ptr: 'int', dtype: 'np.dtype') -> None: <div style="text-align: right; float: right; color: #215f11">method</div> 
```
(No document)
```
### channel\_first <div style="text-align: right; float: right; color: #21138d">property</div>  
```
Indicate whether the buffer is channel-first.
```
### create(width: 'int', height: 'int', channels: 'int' = 3, channel_first: 'bool' = False, device: 'Device | None' = None, use_cupy: 'bool' = False, dtype: 'object' = <class 'numpy.float32'>, channel_order: 'ChannelOrder | str | None' = None) -> "'Buffer'": <div style="text-align: right; float: right; color: #215f11">method</div> 
```
Create a new buffer with allocated storage.
```
### from\_array(device: 'Device', array: 'object', *, channel_order: 'ChannelOrder | str | None' = None, channel_first: 'bool' = False) -> "'Buffer'": <div style="text-align: right; float: right; color: #215f11">method</div> 
```
Wrap an existing array buffer.
```
### load(device: 'Device', source: 'object', normalize: 'bool', copy_data: 'bool' = True, *, channel_order: 'ChannelOrder | str | None' = None, channel_first: 'bool' = False, use_cupy: 'bool' = False) -> "'Buffer'": <div style="text-align: right; float: right; color: #215f11">method</div> 
```
Create a buffer from a PIL image, numpy array, or torch tensor.
```
### release(self) -> 'None': <div style="text-align: right; float: right; color: #215f11">method</div> 
```
Buffer data is owned by the backing array; no explicit release required.
```
### to\_array(self) -> 'np.ndarray': <div style="text-align: right; float: right; color: #215f11">method</div> 
```
Returns a numpy.ndarray copy of the buffer when possible.
```
### to\_tensor(self) -> 'object': <div style="text-align: right; float: right; color: #215f11">method</div> 
```
Returns the backing torch.Tensor when available.
```
