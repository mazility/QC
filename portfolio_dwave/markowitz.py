import numpy as np

class Markowitz:
    n = 0
    G = np.zeros((n, n))
    slope_factor = 3.0
    overall_factor = 0.1
    displacement = -0.9
    dt_const = 0.07
    h = np.zeros(n)
    qmatrix = np.zeros((n, n))
    jmatrix = np.zeros((n, n))
    cfactor = 0
    gfactor = 0
    hvector = np.zeros(n)
    qvector = np.zeros(n)
    theta = [0.3, 0.3, 0.3]
    budget = 100.0

    def to_qubo(self, file_prices, file_averages, file_covariance):
        f = open(file_prices, 'r')
        price = [float(x) for x in f.readline().split(',')]
        f.close()

        f = open(file_averages, 'r')
        avg = [float(x) for x in f.readline().split(',')]
        f.close()

        n = len(avg)
        cov = np.zeros((n, n))
        f = open(file_covariance, 'r')
        i = 0
        for line in f:
            values = line.split(',')
            for j in range(len(values)):
                cov[i, j] = float(values[j])
            i += 1
        f.close()

        qmatrix = np.zeros((n, n))
        qvector = np.zeros((n))
        for row in range(n):
            for col in range(row + 1, n, 1):
                qmatrix[row][col] = 2.0 * (self.theta[1] * cov[row][col] + \
                    self.theta[2] * price[row] * price[col])

        for row in range(n):
            qvector[row] = -1.0 * self.theta[0] * avg[row] - \
            self.theta[2] * 2.0 * self.budget * price[row] + \
            self.theta[1] * cov[row][row] + self.theta[2] * price[row] * price[row]

        cfactor = self.theta[2] * self.budget * self.budget

        return (qmatrix, qvector, cfactor)

    def to_ising(self, qmatrix, qvector, cfactor):
        n = qmatrix.shape[0]
        hvector = np.zeros(n)
        jmatrix = 0.25 * qmatrix

        linear_offset = 0.0
        quadratic_offset = 0.0

        for i in range(n):
            hvector[i] = 0.5 * qvector[i]
            linear_offset += qvector[i]

        for row in range(n):
            for col in range(row + 1, n, 1):
                hvector[row] += 0.25 * qmatrix[row][col]
                hvector[col] += 0.25 * qmatrix[row][col]
                quadratic_offset += qmatrix[row][col]

        gfactor = cfactor + 0.5 * linear_offset + 0.25 * quadratic_offset
        return (jmatrix, hvector, gfactor)

    def energy_qubo(self, spins, add_cfactor = True):
        res = np.dot(np.dot(spins, self.qmatrix), spins) + np.sum(spins * self.qvector)
        if add_cfactor == True:
            res += self.cfactor
        return res

    def energy_ising(self, spins, add_cfactor = False):
        res = np.dot(np.dot(spins, self.jmatrix), spins) + np.sum(spins * self.hvector)
        if add_cfactor == True:
            res += self.gfactor
        return res

    def save_ising(self, filename):
        f = open(filename, 'w')
        f.write("%f\n" % self.slope_factor)
        f.write("%f\n" % self.overall_factor)
        f.write("%f\n" % self.displacement)
        f.write("%f\n" % self.dt_const)
        f.write("%d\n" % self.n)
        for i in range(self.n):
            f.write("%f\n" % self.hvector[i])
        for row in range(self.n):
            for col in range(row + 1, self.n , 1):
                f.write("%d %d %f\n" % (row + 1, col + 1, self.jmatrix[row][col]))

    def save_qubo(self, filename):
        f = open(filename, 'w')
        f.write("%f\n" % self.slope_factor)
        f.write("%f\n" % self.overall_factor)
        f.write("%f\n" % self.displacement)
        f.write("%f\n" % self.dt_const)
        f.write("%d\n" % self.n)
        for i in range(self.n):
            f.write("%f\n" % self.qvector[i])
        for row in range(self.n):
            for col in range(row + 1, self.n , 1):
                f.write("%d %d %f\n" % (row + 1, col + 1, self.qmatrix[row][col]))

    def eval_portfolio(self, spins):
        return 0

    def __init__(self, file_prices, file_averages, file_covariance, theta = [0.3, 0.3, 0.3], budget = 100.0, add_bias = False):
        self.theta = theta
        self.budget = budget
        (qmatrix, qvector, cfactor) = self.to_qubo(file_prices, file_averages, file_covariance)
        (jmatrix, hvector, gfactor) = self.to_ising(qmatrix, qvector, cfactor)
        self.qmatrix = qmatrix
        self.jmatrix = jmatrix
        self.cfactor = cfactor
        self.gfactor = gfactor
        self.n = jmatrix.shape[0]
        if (add_bias == True):
            self.G = jmatrix + hvector
        else:
            self.G = jmatrix
        self.h = np.expand_dims(hvector, axis=0).T
        self.hvector = hvector
        self.qvector = qvector
