"""Paper §5: independent-trajectory torque prediction and RMSE reporting."""
import numpy as np


def rmse_by_actuator(pred, reference):
    pred, ref = np.asarray(pred,float),np.asarray(reference,float)
    if pred.shape!=ref.shape or pred.ndim!=2 or pred.shape[1]!=6:
        raise ValueError("Prediction and reference must both be [N,6]")
    return np.sqrt(np.mean((pred-ref)**2,axis=0))


def compare_predictions(Ytest, torque_test, alternatives):
    """alternatives: dictionary of name -> full parameter or (matrix, coeff) pair.

    This helper does not choose or silently reuse identification data.
    """
    output={}
    for label,theta in alternatives.items():
        prediction=np.einsum('nij,j->ni',Ytest,np.asarray(theta,float))
        output[label]=dict(rmse=rmse_by_actuator(prediction,torque_test).tolist())
    return output
