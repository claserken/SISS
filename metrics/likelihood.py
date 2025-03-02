from metrics.song_likelihood.likelihood import get_likelihood_fn

class LikelihoodEvaluator:
    def __init__(self, sde):
        self.likelihood_fn = get_likelihood_fn(sde)

    def evaluate_likelihood(self, model, img_batch):
        # img_batch should be processed before passing
        return self.likelihood_fn(model, img_batch)
    


        
        