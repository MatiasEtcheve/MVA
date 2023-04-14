import matplotlib.colors as pltcolors
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import optimize
from scipy.linalg import cho_factor, cho_solve
from sklearn import discriminant_analysis, linear_model, metrics, svm


def plotHyperSurface(
    ax, xRange, model, intercept, label, color="grey", linestyle="-", alpha=1.0
):
    # xx = np.linspace(-1, 1, 100)
    if model.type == "linear":
        xRange = np.array(xRange)
        yy = -(model.w[0] / model.w[1]) * xRange - intercept / model.w[1]
        ax.plot(xRange, yy, color=color, label=label, linestyle=linestyle)
    else:
        # import pdb
        # pdb.set_trace()
        xRange = np.linspace(xRange[0], xRange[1], 100)
        X0, X1 = np.meshgrid(xRange, xRange)
        xy = np.vstack([X0.ravel(), X1.ravel()]).T
        Y30 = model.separating_function(xy).reshape(X0.shape) + intercept
        CS = ax.contour(
            X0, X1, Y30, colors=color, levels=[0.0], alpha=alpha, linestyles=[linestyle]
        )
        # CS.collections[0].set_label(label)


def plotClassification(
    X,
    y,
    model=None,
    label="",
    separatorLabel="Separator",
    ax=None,
    bound=[[-1.0, 1.0], [-1.0, 1.0]],
):
    """Plot the SVM separation, and margin"""
    colors = ["blue", "red"]
    labels = [1, -1]
    cmap = pltcolors.ListedColormap(colors)
    if ax is None:
        fig, ax = plt.subplots(1, figsize=(11, 7))
    for k, label in enumerate(labels):
        im = ax.scatter(
            X[y == label, 0], X[y == label, 1], alpha=0.5, label="class " + str(label)
        )

    if model is not None:
        xx = np.array(bound[0])
        plotHyperSurface(ax, bound[0], model, model.b, separatorLabel)
        # Plot margin
        if model.support is not None:
            ax.scatter(
                model.support[:, 0],
                model.support[:, 1],
                label="Support",
                s=80,
                facecolors="none",
                edgecolors="r",
                color="r",
            )
            print("Number of support vectors = %d" % (len(model.support)))
            signedDist = model.separating_function(model.support)
            plotHyperSurface(
                ax,
                xx,
                model,
                -np.min(signedDist),
                "Margin -",
                linestyle="-.",
                alpha=0.8,
            )
            plotHyperSurface(
                ax,
                xx,
                model,
                -np.max(signedDist),
                "Margin +",
                linestyle="--",
                alpha=0.8,
            )

            margin = (np.max(signedDist) - np.min(signedDist)) / model.norm_f
            ax.set_title("Margin = %.3f" % (margin))

            # Plot points on the wrong side of the margin
            totalsignedDist = model.separating_function(X)
            supp_min = X[(totalsignedDist > np.min(signedDist)) * (y == -1)]
            supp_max = X[(totalsignedDist < np.max(signedDist)) * (y == 1)]
            wrong_side_points = np.concatenate([supp_min, supp_max], axis=0)

            ax.scatter(
                wrong_side_points[:, 0],
                wrong_side_points[:, 1],
                label="Beyond the margin",
                s=80,
                facecolors="none",
                edgecolors="grey",
                color="grey",
            )

    ax.legend(loc="upper left")
    ax.grid()
    ax.set_xlim(bound[0])
    ax.set_ylim(bound[1])


def plotRegression(
    X,
    Y,
    Y_clean=None,
    model=None,
    label="",
    regressionLabel="Regression",
    ax=None,
    bound=[[-1.0, 1.0], [-0.8, 1.2]],
):
    """Plot the SVM separation, and margin"""
    colors = ["blue"]
    cmap = pltcolors.ListedColormap(colors)
    if ax is None:
        fig, ax = plt.subplots(1, figsize=(11, 7))
    im = ax.scatter(X, Y, alpha=0.5, label="data")
    if Y_clean is not None:
        ax.plot(X, Y_clean, color="green", label="ground truth", linestyle="-")

    if model is not None:
        xx = np.linspace(bound[0][0], bound[0][1], 100)
        Y_pred = model.regression_function(X.reshape(-1, 1))
        ax.plot(
            xx, Y_pred + model.b, color="grey", label=regressionLabel, linestyle="-"
        )
        # Plot margin
        if model.type == "svr" and model.support is not None:
            ind = (Y - Y_pred > model.b + model.eta) + (
                Y - Y_pred < model.b - model.eta
            )
            ax.scatter(
                X[ind],
                Y[ind],
                label="Beyond the margin",
                s=80,
                facecolors="none",
                edgecolors="grey",
                color="grey",
            )
            ax.scatter(
                model.support[:, 0],
                model.support[:, 1],
                label="Support",
                s=80,
                facecolors="none",
                edgecolors="r",
                color="r",
            )
            print("Number of support vectors = %d" % (len(model.support)))
            ax.plot(
                xx,
                Y_pred + model.b + model.eta,
                color="grey",
                label="Tube +",
                linestyle="-.",
                alpha=0.8,
            )
            ax.plot(
                xx,
                Y_pred + model.b - model.eta,
                color="grey",
                label="Tube -",
                linestyle="--",
                alpha=0.8,
            )

        # Plot points outside the tube

    ax.legend(loc="upper left")
    ax.grid()
    ax.set_xlim(bound[0])
    ax.set_ylim(bound[1])


def scatter_label_points(X, y, ax=None, title=""):
    if ax is None:
        fig, ax = plt.subplots(1, figsize=(11, 7))
    colormap = np.array(["r", "g", "b"])
    ax.scatter(X[:, 0], X[:, 1], s=200, c=colormap[y], alpha=0.5)
    ax.set_title(title)


def plot_multiple_images(images, num_row=1, num_col=10):
    fig, axes = plt.subplots(num_row, num_col, figsize=(num_col, num_row))
    num = num_row * num_col
    for i in range(num):
        if num_row > 1:
            ax = axes[i // num_col, i % num_col]
        else:
            ax = axes[i]
        ax.imshow(images[i], cmap="gray")
        ax.set_axis_off()
    plt.tight_layout()
    # plt.show()


def gaussian_mixture(N, mu=0.3, sigma=0.1):
    """Mixture of two gaussians"""
    X = np.random.normal(mu, sigma, (N, 2))
    u = np.random.uniform(0, 1, N) > 0.5
    Y = 2.0 * u - 1
    X *= Y[:, np.newaxis]
    X -= X.mean(axis=0)
    return X, Y


def generateXor(n, mu=0.5, sigma=0.5):
    """Four gaussian clouds in a Xor fashion"""
    X = np.random.normal(mu, sigma, (n, 2))
    yB0 = np.random.uniform(0, 1, n) > 0.5
    yB1 = np.random.uniform(0, 1, n) > 0.5
    # y is in {-1, 1}
    y0 = 2.0 * yB0 - 1
    y1 = 2.0 * yB1 - 1
    X[:, 0] *= y0
    X[:, 1] *= y1
    X -= X.mean(axis=0)
    return X, y0 * y1


def generateMexicanHat(N, stochastic=False):
    xMin = -1
    xMax = 1.0
    sigma = 0.2
    std = 0.1
    if stochastic:
        x = np.random.uniform(xMin, xMax, N)
    else:
        x = np.linspace(xMin, xMax, N)
    yClean = (1 - x**2 / sigma**2) * np.exp(-(x**2) / (2 * sigma**2))
    y = yClean + np.random.normal(0, std, N)
    return x, y, yClean


def generateRings(N):

    N_rings = 3
    Idex = [0, int(N / 3), int(2 * N / 3), N]
    std = 0.1

    Radius = np.array([1.0, 2.0, 3.0])
    y = np.ones(N)

    X = np.random.normal(size=(N, 2))
    X = np.einsum("ij,i->ij", X, 1.0 / np.sqrt(np.sum(X**2, axis=1)))
    for i in range(N_rings):
        X[Idex[i] : Idex[i + 1], :] *= Radius[i]
        y[Idex[i] : Idex[i + 1]] = i
    y = y.astype(int)
    return X, y


def loadMNIST(path):
    import gzip

    a_file = gzip.open(path, "rb")
    N = 2000
    image_size = 28
    num_images = 2 * N
    buf = a_file.read(image_size * image_size * num_images)
    data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
    data = data.reshape(num_images, image_size, image_size)
    data = data / 255.0

    clean_train, clean_test = data[:N], data[N:]
    train = clean_train + np.random.normal(loc=0.0, scale=0.5, size=clean_train.shape)
    test = clean_test + np.random.normal(loc=0.0, scale=0.5, size=clean_test.shape)

    data = {
        "cleanMNIST": {"train": clean_train, "test": clean_test},
        "noisyMNIST": {"train": train, "test": test},
    }
    return data
