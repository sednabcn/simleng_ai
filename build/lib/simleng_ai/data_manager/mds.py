
class MDS:

    def mds():
        # Pairwise distance between European cities
        try:
            url = '../data/eurodist.csv'
            df = pd.read_csv(url)
        except:
            url ='https://raw.githubusercontent.com/neurospin/pystatsml/master/datasets/eurodist.csv'
        df = pd.read_csv(url)
        print(df.ix[:5, :5])
        city = df["city"]
        D = np.array(df.ix[:, 1:])
        # Distance matrix
        # Arbitrary choice of K=2 components
        skmds = skMDS(dissimilarity='precomputed', n_components=2, random_state=40,\
                      max_iter=3000,eps=1e-9)
        X = skmds.fit_transform(D)
        #Recover coordinates of the cities in Euclidean referential whose orientation is arbitrary:

        Deuclidean = metrics.pairwise.pairwise_distances(X, metric='euclidean')
        print(np.round(Deuclidean[:5, :5]))

        #Plot the results:
        # Plot: apply some rotation and flip
        theta = 80 * np.pi / 180.
        rot = np.array([[np.cos(theta), -np.sin(theta)],
                        [np.sin(theta), np.cos(theta)]])
        Xr = np.dot(X, rot)
        # flip x
        Xr[:, 0] *= -1
        plt.scatter(Xr[:, 0], Xr[:, 1])
        for i in range(len(city)):
            plt.text(Xr[i, 0], Xr[i, 1], city[i])
        plt.axis('equal')
        plt.show()
        #Determining the number of components
        #We must choose K * ∈ {1, . . . , K} the number of required components.
        #Plotting the values of the stress function, obtained using k ≤ N − 1 components.
        #In general, start with 1, . . . K ≤ 4. Choose K * where you can clearly distinguish
        #an elbow in the stress curve.
        #Thus, in the plot below, we choose to retain information accounted for by the first
        #two components, since this is where the elbow is in the stress curve.
        k_range = range(1, min(5, D.shape[0]-1))
        stress = [skMDS(dissimilarity='precomputed', n_components=k,
                      random_state=42, max_iter=300, eps=1e-9).fit(D).stress_ for k in k_range]
        print(stress)
        plt.plot(k_range, stress)
        plt.xlabel("k")
        plt.ylabel("stress")
        plt.show()
    def non_linear_mds():
      #!/usr/bin/env python
      #Nonlinear dimensionality reduction
      #Sources:
      #• Scikit-learn documentation
      #• Wikipedia
      """
      Nonlinear dimensionality reduction or manifold learning cover unsupervised methods 
      that attempt to identify low-dimensional manifolds within the original
      P -dimensional space that represent high data density. Then those methods
      provide a mapping from the high-dimensional space to the low-dimensional embedding.
      Isomap
      Isomap is a nonlinear dimensionality reduction method that combines a procedure
      to compute the distance matrix with MDS. The distances calculation is based on
      geodesic distances evaluated on neighborhood graph:
      1. Determine the neighbors of each point. All points in some fixed radius
      or K nearest neighbors.
      2. Construct a neighborhood graph. Each point is connected to other if
      it is a K nearest neighbor. Edge length equal to Euclidean distance.
      3. Compute shortest path between pairwise of points d ij to build the distance matrix D.
      4. Apply MDS on D.
      """

      X, color = datasets.samples_generator.make_s_curve(1000, random_state=42)
      fig = plt.figure(figsize=(10, 5))
      plt.suptitle("Isomap Manifold Learning", fontsize=14)
      ax = fig.add_subplot(121, projection='3d')
      ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap=plt.cm.Spectral)
      ax.view_init(4, -72)
      plt.title('2D "S shape" manifold in 3D')
      Y = manifold.Isomap(n_neighbors=10, n_components=2).fit_transform(X)
      ax = fig.add_subplot(122)
      plt.scatter(Y[:, 0], Y[:, 1], c=color, cmap=plt.cm.Spectral)
      plt.title("Isomap")
      plt.xlabel("First component")
      plt.ylabel("Second component")
      plt.axis('tight')
      plt.show()
