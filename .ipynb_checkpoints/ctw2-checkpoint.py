import numpy as np

class CTW:
    def __init__(self, depth, symbols=2, sidesymbols=1, staleness=0, alpha=None):
        """
        depth:       The tree depth (D).
        symbols:     Number of possible symbols (M).
        sidesymbols: Number of possible side-info states.
        staleness:   Ignore the last K side-info states if > 0.
        alpha:       A 1D array/list of length M specifying Dirichlet prior params.
                     If None, we default to 0.5 for each symbol (KT style).
        """
        self.D = depth
        self.M = symbols
        self.K = staleness
        # If alpha isn't specified, use 0.5 for all symbols (Krichevsky–Trofimov default)
        if alpha is None:
            alpha = [0.5]*symbols
        self.alpha = np.array(alpha, dtype=float)

        # total number of (symbol, side-symbol) states if side info is used
        self.Mtot = symbols*sidesymbols

        # "restricted" contexts (no side info)
        self.rcontexts = range(symbols)
        # "complete" contexts (with side info)
        self.ccontexts = []
        for x in range(symbols):
            for y in range(sidesymbols):
                self.ccontexts.append((x,y))

        # store leaf nodes in a dict
        self.leaves = {}

        # build the tree starting from root
        self.root = Node(parent=None, context=[], ctw=self)

        # distribution at the root (start uniform or prior-based)
        # We'll keep it uniform for initialization
        self.distribution = np.ones((symbols,)) / symbols

    def set_distribution(self, distribution):
        self.distribution = distribution

    def get_distribution(self):
        return self.distribution

    def add_leaf(self, node):
        self.leaves[str(node.context)] = node

    def update(self, symbol, context):
        leaf = self.leaves[str(context)]
        leaf.update(symbol)

    def predict_sequence(self, seq, sideseq=None):
        """
        seq:     main sequence (length N).
        sideseq: side info sequence (length N) or None.
        Returns:
          A 2D array of shape (M, N - D), giving the distribution at each step.
        """
        N = len(seq)
        if N <= self.D:
            raise ValueError("Sequence length must exceed depth.")

        distributions = np.zeros((self.M, N - self.D))

        # build initial context of length D
        if sideseq is None:
            context = [seq[d] for d in reversed(range(self.D))]
        else:
            ccontext = [(seq[d], sideseq[d]) for d in reversed(range(self.D))]
            context = ccontext.copy()
            if self.K > 0:
                for k in range(self.K):
                    context[k] = ccontext[k][0]

        # loop through the sequence from index D onward
        for n, x in enumerate(seq[self.D:]):
            # update the leaf node for the current context
            self.update(x, context)
            # record the root distribution
            distributions[:, n] = self.get_distribution()

            # shift the context by one
            if sideseq is None:
                context.insert(0, x)
                context = context[:self.D]
            else:
                ccontext.insert(0, (x, sideseq[n+self.D]))
                ccontext = ccontext[:self.D]
                context = ccontext.copy()
                if self.K > 0:
                    for k in range(self.K):
                        context[k] = ccontext[k][0]

        return distributions


class Node:
    def __init__(self, ctw, context, parent):
        self.ctw = ctw
        self.counts = np.zeros((self.ctw.M,), dtype=float)
        self.context = context
        self.parent = parent
        self.beta = 1.0

        # determine if root or leaf
        self.root = (len(self.context) == 0)
        if len(self.context) == self.ctw.D:
            self.leaf = True
            self.ctw.add_leaf(self)
        else:
            self.leaf = False

        # create children if not a leaf
        if not self.leaf:
            # decide if we branch on side info or not
            if (self.ctw.Mtot > self.ctw.M) and (len(self.context) >= self.ctw.K):
                contexts = self.ctw.ccontexts
            else:
                contexts = self.ctw.rcontexts
            self.children = []
            for c in contexts:
                self.children.append(Node(
                    ctw=ctw,
                    context=context + [c],
                    parent=self
                ))

    def update(self, symbol, etain=None):
        """
        Incorporates one observed symbol at this node.
        'etain' is the incoming distribution from the child. If None,
        we build it from our local Dirichlet-based distribution.
        """
        alpha = self.ctw.alpha  # Dirichlet prior parameters, shape [M]
        M = self.counts.size

        # If there's no incoming distribution from a deeper node,
        # we create a partial distribution 'etain' for the first M-1 classes,
        # using the local Dirichlet posterior.
        if etain is None:
            local_pe = (self.counts + alpha) / (np.sum(self.counts) + np.sum(alpha))
            # The code expects 'etain' to be the ratio of the first M-1 probabilities
            # to the last probability. i.e. etain[i] = local_pe[i]/local_pe[M-1]
            # for i in [0..M-2].
            # This is how the original code merges child vs. parent distributions.
            if M > 1:
                etain = local_pe[:-1] / local_pe[-1]
            else:
                # If M=1 (rare but let's handle), it's trivially 1
                etain = np.array([1.0])

        # 'pw' is the distribution that merges the child mixture with one extra slot
        pw = np.append(etain, 1.0)
        pw /= np.sum(pw)

        # Local Dirichlet (non-symmetric if alpha != 0.5) estimate
        pe = (self.counts + alpha) / (np.sum(self.counts) + np.sum(alpha))

        # Build the outgoing distribution 'etaout' similarly:
        # For the first M-1 symbols:
        #   etaout[i] = (self.beta * pe[i] + pw[i]) / (self.beta * pe[-1] + pw[-1])
        # This merges local vs. child distribution in CTW manner.
        etaout = (self.beta * pe[:-1] + pw[:-1]) / (self.beta * pe[-1] + pw[-1])

        # Update the weight ratio beta based on how well local vs child predicted 'symbol'
        self.beta *= pe[symbol] / pw[symbol]

        # Increment the count for the observed symbol
        self.counts[symbol] += 1.0

        # Pass the new distribution up the tree or set the root distribution
        if not self.root:
            self.parent.update(symbol, etaout)
        else:
            # At the root, finalize the CTW distribution
            etasum = np.sum(etaout) + 1.0
            root_dist = np.append(etaout, 1.0) / etasum
            self.ctw.set_distribution(root_dist)
