import numpy as np

class CTW:
    def __init__(self, depth, symbols=2, sidesymbols=1, sideseqtwo=None, staleness=0, alpha=None):
        """
        depth:       The tree depth (D).
        symbols:     Number of possible main symbols (M).
        sidesymbols: Number of possible states for the first side-info sequence.
        sideseqtwo:  Either:
                       - A number representing the number of possible states for the second side-info sequence, or
                       - A list/tuple of values, from which the number of unique states is computed.
        staleness:   Ignore the last K side-info states if > 0.
        alpha:       A 1D array/list of length M specifying Dirichlet prior parameters.
                     If None, we default to 0.5 for each symbol (KT style).
        """
        self.D = depth
        self.M = symbols
        self.K = staleness

        if alpha is None:
            alpha = [0.5] * symbols
        self.alpha = np.array(alpha, dtype=float)

        # Process sideseqtwo: allow either a number or a list/tuple.
        if isinstance(sideseqtwo, (list, tuple)):
            # Treat the provided list as the full set of side-information values.
            self.sideseqtwo_flag = True
            # Determine number of unique states from the list.
            unique_states = set(sideseqtwo)
            self.sideseqtwo_states = len(unique_states)
            self.sideseqtwo = self.sideseqtwo_states
        elif sideseqtwo is not None:
            self.sideseqtwo_flag = True
            self.sideseqtwo = sideseqtwo
        else:
            self.sideseqtwo_flag = False

        # Calculate total number of states in complete contexts.
        if self.sideseqtwo_flag:
            self.Mtot = symbols * sidesymbols * self.sideseqtwo
        else:
            self.Mtot = symbols * sidesymbols

        # "Restricted" contexts (only main symbols)
        self.rcontexts = range(symbols)
        # "Complete" contexts (with side info)
        if self.sideseqtwo_flag:
            # Each complete context is a triple: (main, side1, side2)
            self.ccontexts = [(x, y, z) for x in range(symbols)
                                        for y in range(sidesymbols)
                                        for z in range(self.sideseqtwo)]
        else:
            # Each complete context is a pair: (main, side1)
            self.ccontexts = [(x, y) for x in range(symbols) for y in range(sidesymbols)]

        # Dictionary to hold leaf nodes
        self.leaves = {}

        # Build the tree starting from the root
        self.root = Node(ctw=self, context=[], parent=None)

        # Initialize the distribution at the root (uniform or prior-based)
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

    def predict_sequence(self, seq, sideseq=None, sideseqtwo=None):
        """
        seq:         Main sequence (length N).
        sideseq:     First side info sequence (length N) or None.
        sideseqtwo:  Second side info sequence (length N) or None.
                     When using two side sequences, sideseqtwo must be provided.
        Returns:
          A 2D array of shape (M, N - D), giving the predictive distribution at each step.
        """
        N = len(seq)
        if N <= self.D:
            raise ValueError("Sequence length must exceed depth.")

        distributions = np.zeros((self.M, N - self.D))

        # Build the initial context of length D
        if sideseq is None:
            context = [seq[d] for d in reversed(range(self.D))]
        else:
            if self.sideseqtwo_flag:
                if sideseqtwo is None:
                    raise ValueError("CTW is configured for two side sequences but sideseqtwo is None.")
                # For two side sequences, each context element is a triple.
                ccontext = [(seq[d], sideseq[d], sideseqtwo[d]) for d in reversed(range(self.D))]
            else:
                # For one side sequence, each context element is a pair.
                ccontext = [(seq[d], sideseq[d]) for d in reversed(range(self.D))]
            context = ccontext.copy()
            if self.K > 0:
                for k in range(self.K):
                    # Replace side-info with just the main symbol
                    context[k] = ccontext[k][0]

        # Process the sequence from index D onward.
        for n, x in enumerate(seq[self.D:]):
            # Update the leaf node corresponding to the current context.
            self.update(x, context)
            # Record the predictive distribution at the root.
            distributions[:, n] = self.get_distribution()

            # Shift the context by one position.
            if sideseq is None:
                context.insert(0, x)
                context = context[:self.D]
            else:
                if self.sideseqtwo_flag:
                    ccontext.insert(0, (x, sideseq[n + self.D], sideseqtwo[n + self.D]))
                else:
                    ccontext.insert(0, (x, sideseq[n + self.D]))
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

        # Determine if this node is the root or a leaf.
        self.root = (len(self.context) == 0)
        if len(self.context) == self.ctw.D:
            self.leaf = True
            self.ctw.add_leaf(self)
        else:
            self.leaf = False

        # Create children if not a leaf.
        if not self.leaf:
            if (self.ctw.Mtot > self.ctw.M) and (len(self.context) >= self.ctw.K):
                contexts = self.ctw.ccontexts
            else:
                contexts = self.ctw.rcontexts
            self.children = []
            for c in contexts:
                self.children.append(Node(
                    ctw=self.ctw,
                    context=context + [c],
                    parent=self
                ))

    def update(self, symbol, etain=None):
        """
        Incorporates one observed symbol at this node.
        'etain' is the incoming distribution from a deeper node.
        If None, we compute it using the local Dirichlet-based estimate.
        """
        alpha = self.ctw.alpha  # Dirichlet prior parameters
        M = self.counts.size

        if etain is None:
            local_pe = (self.counts + alpha) / (np.sum(self.counts) + np.sum(alpha))
            if M > 1:
                etain = local_pe[:-1] / local_pe[-1]
            else:
                etain = np.array([1.0])

        pw = np.append(etain, 1.0)
        pw /= np.sum(pw)
        pe = (self.counts + alpha) / (np.sum(self.counts) + np.sum(alpha))
        etaout = (self.beta * pe[:-1] + pw[:-1]) / (self.beta * pe[-1] + pw[-1])

        self.beta *= pe[symbol] / pw[symbol]
        self.counts[symbol] += 1.0

        if not self.root:
            self.parent.update(symbol, etaout)
        else:
            etasum = np.sum(etaout) + 1.0
            root_dist = np.append(etaout, 1.0) / etasum
            self.ctw.set_distribution(root_dist)
