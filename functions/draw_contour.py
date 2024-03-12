import numpy as np
import matplotlib.pyplot as plt


def draw_contour(KX, KY, conduction_E, valence_E, conduction=True, surf=False, diff=False):

    # 3D plot
    if surf:
        plt.figure(dpi=100)
        ax = plt.axes(projection='3d')
        cmap = plt.get_cmap('coolwarm')
        ax.plot_surface(KX, KY, conduction_E, cmap=cmap)
        ax.plot_surface(KX, KY, valence_E, cmap=cmap)
        plt.show()

    # contour plot of the difference to show gaps
    if diff:
        plt.figure(dpi=100)
        cmap = plt.get_cmap('jet')

        arr = conduction_E - valence_E
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                if arr[i, j] > 0.01:
                    arr[i, j] = np.NaN

        plt.contourf(KX, KY, arr, 50, cmap=cmap)
        plt.colorbar()
    else:
        # contour plot
        plt.figure(dpi=100)
        cmap = plt.get_cmap('coolwarm')
        if conduction:
            plt.contourf(KX, KY, conduction_E, 50, cmap=cmap)
        else:
            plt.contourf(KX, KY, valence_E, 50, cmap=cmap)
        plt.colorbar()
