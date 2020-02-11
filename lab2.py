import numpy as np

class Convolution3D():
    def __init__(self, kernel):
        self.kernel = kernel

    def control(point, range):
        return 0 <= point[0] < range[0] and 0 <= point[1] < range[1]
        
    def clamp (value, min, max):
        return np.maximum(np.minimum(value, max), min)

    def calculation(self, input_matrix):
        w, h, K = self.kernel.shape
        W, H, K = input_matrix.shape
        
        output_matrix = np.zeros((W, H), dtype="uint8")
        
        for x, y in np.ndindex(W, H):
            pixel = 0
            
            for x_, y_, z_ in np.ndindex(w, h, K):
                tx = x - w // 2 + x_
                ty = y - h // 2 + y_

                if control((tx, ty), (W, H)):
                    pixel += input_matrix[tx][ty][z_] * self.kernel[x_][y_][z_]
            
            output_matrix[x][y] = clamp(pixel, 0, 255)
                    
        return output_matrix