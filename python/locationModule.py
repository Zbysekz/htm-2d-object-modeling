import matplotlib.pyplot as plt
from GCM_1D import GCM_1D

gcm = [GCM_1D(size=50, scale=1.0) for x in range(0, 10)]

for g in gcm:
    g.InitRandomPos()

bins = range(0, 50, 1)

fig, _ = plt.subplots(nrows=len(gcm), ncols=1, figsize=(6, 4))

for i in range(50):

    idAxe = 0
    for g in gcm:
        g.Compute()

        #g.Shift(2.0)

        g.plot(fig.axes[idAxe])

        idAxe += 1
    fig.axes[idAxe-1].get_xaxis().set_visible(True)


    fig.canvas.draw()
    plt.show(block=False)
    plt.pause(0.05)  # delay is needed for proper redraw

    fig.axes[0].clear()

    #gaussian function (normal distribution)
    #plt.plot(bins, gcm.cellActivities,
             #linewidth=2, color='r')
    #plt.show()

