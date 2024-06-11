import numpy as np
import matplotlib.ticker
import matplotlib.pyplot as plt

# converted to MeV

def draw_contour(KX, KY, conduction_E, valence_E, conduction=True, surf=False, diff=False, cut_off=0.01):
    cm = 1 / 2.54
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
        plt.figure(figsize=(10*cm, 7.5*cm), dpi=600)
        cmap = plt.get_cmap('coolwarm')
        arr = conduction_E - valence_E
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                if arr[i, j] > cut_off:
                    arr[i, j] = np.NaN
        plt.contourf(KX, KY, arr*1000, levels=1000, vmin=0, vmax=10, cmap=cmap)
        clb = plt.colorbar(format=matplotlib.ticker.FormatStrFormatter('%.1f')) #, ticks=[0, 2, 4, 6, 8, 10])
        clb.ax.set_title('meV')
    else:
        # contour plot
        plt.figure(figsize=(20*cm, 7.5*2*cm), dpi=100)
        cmap = plt.get_cmap('coolwarm')
        if conduction:
            plt.contourf(KX, KY, conduction_E, 50, cmap=cmap)
        else:
            plt.contourf(KX, KY, valence_E, 50, cmap=cmap)
        plt.colorbar()
