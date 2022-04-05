# 1개 배열 로딩
np.load("./array1.npy")
>array([[8, 7, 4],
      [4, 3, 8]])
      
# 복수 파일 로딩
npzfiles = np.load("./array2.npz")
npzfiles.files
> ['arr_0', 'arr_1']

npzfiles['arr_0']
>array([[8, 7, 4],
      [4, 3, 8]])
      
npzfiles['arr_1']
>array([[6, 9, 8],
      [9, 7, 7]])
