class ModelArgs():
    def __init__(self, bpdecay=0.1, l1_weight=0.0, l2_weight=1.0, elbo_weight=0.1, resamp_alpha=0.3):
        self.bpdecay = bpdecay
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight
        self.elbo_weight = elbo_weight
        self.resamp_alpha = resamp_alpha