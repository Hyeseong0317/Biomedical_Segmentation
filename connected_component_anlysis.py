import numpy as np
import matplotlib.pyplot as plt


connectivity_8 = np.array([[1,1,1],
                           [1,0,1],
                           [1,1,1]])

def find_labels(labels, r, c, neighbors):
    
    tmp_labels = labels[r-1:r+2, c-1:c+2]*neighbors
    
    return np.sort(tmp_labels[np.nonzero(tmp_labels)])


def connected_component_labeling(bin_img, connectivity=connectivity_8):
    equivalent = []
    labels = np.zeros_like(bin_img, dtype='int64')
    next_label = 1
    
    # 1st pass
    for r, row in enumerate(bin_img):
        for c, pixel in enumerate(row):
        
            if pixel!=0:
                neighbors = bin_img[r-1:r+2, c-1:c+2]*connectivity
                num_neighbors = np.count_nonzero(neighbors)
                
                if num_neighbors == 0:
                    labels[r,c] = next_label
                    equivalent.append([next_label,next_label])
                    next_label += 1
                else:
                    L = find_labels(labels, r, c, neighbors)
                    labels[r,c] = np.min(L)
                
                    uni_L = np.unique(L)
                    if len(uni_L)>1:
                        for i, e in enumerate(equivalent):
                            if uni_L[0] in e:
                                equivalent[i].extend(uni_L[1:])
                                equivalent[i] = list(sorted(set(equivalent[i])))
            
# 2nd pass
    for e in equivalent:
        for f in reversed(e):
            labels[labels==f] = e[0]
    return labels
           
def threshold_labels(labels, threshold=10000):
  
  unique_elements, counts_elements = np.unique(labels, return_counts=True)
  thr_elements = unique_elements[counts_elements>threshold]
  
  thr_labels = np.zeros_like(labels)
  
  cnt = 0
  for e in thr_elements:
    if e != 0:
      cnt += 1
      thr_labels[labels==e] = cnt
  return thr_labels
                
            
