class BezierB:
    def __init__(self, points):
        self.points = points

    @staticmethod
    def mid_point(p1, p2):
        return (p1 + p2) / 2

    @staticmethod
    def window(n, seq):
        """
        Returns a sliding window (of width n) over data from the iterable
           s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...
        """
        from itertools import islice
        it = iter(seq)
        result = tuple(islice(it, n))
        if len(result) == n:
            yield result
        for elem in it:
            result = result[1:] + (elem,)
            yield result

    @staticmethod
    def tuck(p0, pl, pr, s):
        laplace = ((pl - p0) + (pr - p0)) / 2
        return p0 + s * laplace

    def refine(self):
        if len(self.points) < 2:
            return
        points = [self.points[0]]
        for p0, p1 in self.window(2, self.points):
            points.extend([self.mid_point(p0, p1), p1])
        self.points = points

    def tuck_all(self, s):
        if len(self.points) < 3:
            return
        points = [self.points[0]]
        for pl, p0, pr in self.window(3, self.points):
            points.append(self.tuck(p0, pl, pr, s))
        points.append(self.points[-1])
        self.points = points

    def smooth(self, smoothness=5):
        for i in range(smoothness):
            self.refine()
            self.tuck_all(1 / 2)
            self.tuck_all(-1)
        return self.points
