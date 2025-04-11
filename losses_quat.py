import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *


## Quaternions
#Computes q^T A q
def quat_self_supervised_primal_loss(q, A, reduce=True):
    losses = torch.einsum('bn,bnm,bm->b', q, A, q)
    # losses = torch.einsum('bijk,ij,bijk->b', q, A, q)


    loss = losses.mean() if reduce else losses
    return loss

def quat_consistency_loss(qs, q_target, reduce=True):
    q = qs[0]
    q_inv = qs[1]

    # Broadcasting to handle different batch sizes
    q = q.unsqueeze(0).expand(q_target.shape[0], -1)
    q_inv = q_inv.unsqueeze(0).expand(q_target.shape[0], -1)

    assert (q.shape == q_inv.shape == q_target.shape)
    d1 = quat_loss(q, q_target, reduce=False)
    d2 = quat_loss(q_inv, quat_inv(q_target), reduce=False)
    d3 = quat_loss(q, quat_inv(q_inv), reduce=False)
    losses =  d1*d1 + d2*d2 + d3*d3
    loss = losses.mean() if reduce else losses
    return loss

def quat_chordal_squared_loss(q, q_target, reduce=True):
    # print('q.shape',q.shape, 'q_target.shape', q_target.shape)
    assert(q.shape == q_target.shape)
    d = quat_norm_diff(q, q_target)
    losses =  2*d*d*(4. - d*d) 
    loss = losses.mean() if reduce else losses
    return loss


def quat_squared_loss(q, q_target, reduce=True):
    assert(q.shape == q_target.shape)
    d = quat_norm_diff(q, q_target)
    losses =  0.5*d*d
    loss = losses.mean() if reduce else losses
    return loss

def quat_loss(q, q_target, reduce=True):
    assert(q.shape == q_target.shape)
    d = quat_norm_diff(q, q_target)
    losses = d*d
    loss = losses.mean() if reduce else losses
    return loss

def quat_squ_loss(q, q_target, reduce=True): #Redhwan
    assert(q.shape == q_target.shape)
    d = quat_norm_diff(q, q_target)
    losses = d
    loss = losses.mean() if reduce else losses
    return loss

#==============================================================================================================
## Rotation matrices
def rotmat_frob_squared_norm_loss(C, C_target, reduce=True):
    """Return the Frobenius norm of the difference betwen two batchs of N rotation matrices."""
    assert(C.shape == C_target.shape)
    if C.dim() < 3:
        C = C.unsqueeze(dim=0)
        C_target = C_target.unsqueeze(dim=0)
    losses = (C - C_target).norm(dim=[1,2])**2 #6. - 2.*trace(C.bmm(C_target.transpose(1,2)))
    loss = losses.mean() if reduce else losses
    return loss


# from here to end by #Redhwan

def opal_loss(q_pred, q_true, alpha=0.1, reduce=True):
    q_pred = F.normalize(q_pred, p=2, dim=-1)
    q_true = F.normalize(q_true, p=2, dim=-1)

    assert q_pred.shape == q_true.shape

    # Calculate quaternion distance matrix
    dist_matrix = quat_norm_diff(q_pred.unsqueeze(1), q_true.unsqueeze(1))

    # Element-wise exponentiation
    opal_loss = torch.exp(-alpha * dist_matrix)

    if reduce:
        opal_loss = opal_loss.mean()

    return opal_loss

def quaternion_to_rotation_matrix1(q):
    """
    Convert quaternion to rotation matrix.

    Parameters:
    - q: Quaternions (batch_size x 4)

    Returns:
    - R: Rotation matrices (batch_size x 3 x 3)
    """
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    R = torch.stack([
        1 - 2 * y ** 2 - 2 * z ** 2, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y,
        2 * x * y + 2 * w * z, 1 - 2 * x ** 2 - 2 * z ** 2, 2 * y * z - 2 * w * x,
        2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x ** 2 - 2 * y ** 2
    ], dim=-1).view(-1, 3, 3)

    return R

def frobenius_squared_norm_loss(q_pred, q_true):
    """
    Frobenius squared norm loss between predicted and true quaternions.

    Parameters:
    - q_pred: Predicted quaternions (batch_size x 4)
    - q_true: True quaternions (batch_size x 4)

    Returns:
    - frobenius_loss: Frobenius squared norm loss
    """

    # Convert quaternions to rotation matrices
    R_pred = quaternion_to_rotation_matrix1(q_pred)
    R_true = quaternion_to_rotation_matrix1(q_true)

    # Calculate Frobenius norm squared loss
    frobenius_loss = torch.norm(R_pred - R_true, p='fro')**2

    return frobenius_loss
#===========================================================================================================================
def bingham_likelihood_loss(predicted_quat, target_quat, concentration_param=1.0): #redhwan
    """
    Bingham likelihood loss for quaternions.

    Args:
    - predicted_quat (torch.Tensor): Predicted quaternions (batch_size x 4).
    - target_quat (torch.Tensor): Target quaternions (batch_size x 4).
    - concentration_param (float): Concentration parameter for the Bingham distribution.

    Returns:
    - loss (torch.Tensor): Bingham likelihood loss.
    """
    # Normalize quaternions
    predicted_quat = predicted_quat / torch.norm(predicted_quat, dim=1, keepdim=True)
    target_quat = target_quat / torch.norm(target_quat, dim=-1, keepdim=True)


    # Bingham likelihood calculation
    dot_product = torch.sum(predicted_quat * target_quat, dim=1)
    loss = -torch.exp(concentration_param * dot_product).mean()

    return loss


def quaternion_norm(q): #Redhwan
    return torch.norm(q, dim=-1, keepdim=True)

def quaternion_conjugate(q): #Redhwan
    return torch.cat([q[:, 0:1], -q[:, 1:]], dim=-1)

def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1.unbind(dim=-1)
    w2, x2, y2, z2 = q2.unbind(dim=-1)
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
    return torch.stack([w, x, y, z], dim=-1)

def bingham_loss(predicted_quat, target_quat, concentration_param=1.0):#Redhwan
    """
    Bingham loss for quaternions.

    Args:
    - predicted_quat (torch.Tensor): Predicted quaternions (batch_size x 4).
    - target_quat (torch.Tensor): Target quaternions (batch_size x 4).
    - concentration_param (float): Concentration parameter for the Bingham distribution.

    Returns:
    - loss (torch.Tensor): Bingham loss.
    """
    # Normalize quaternions
    predicted_quat = predicted_quat / quaternion_norm(predicted_quat)
    target_quat = target_quat / quaternion_norm(target_quat)

    # Bingham loss calculation
    dot_product = torch.sum(predicted_quat * target_quat, dim=-1)
    loss = -torch.exp(concentration_param * dot_product).mean()

    return loss


import torch


def quaternion_dot(q1, q2):
    return torch.sum(q1 * q2, dim=-1)


def quat_geodesic_loss_antipodal(q, q_target, reduce=True):
    """
    Compute the geodesic loss between two sets of antipodal unit quaternions.

    Parameters:
    - q: Tensor representing the predicted quaternions (batch_size x 4).
    - q_target: Tensor representing the target quaternions (batch_size x 4).
    - reduce: If True, compute the mean loss over the batch. If False, return individual losses.

    Returns:
    - Geodesic loss between the predicted and target quaternions.
    """
    assert q.shape == q_target.shape, "Input shapes must be the same"
    assert q.shape[-1] == 4, "Last dimension of quaternions should be 4"

    dot_products = torch.sum(q * q_target, dim=-1)
    dot_products = torch.clamp(dot_products, -1.0, 1.0)  # Ensure dot_products are within the valid range

    angle_diff = 2.0 * torch.acos(torch.abs(dot_products))

    # Handle antipodal symmetry: take the minimum angle between q and -q
    antipodal_dot_products = torch.sum(q * (-q_target), dim=-1)
    antipodal_angle_diff = 2.0 * torch.acos(torch.abs(antipodal_dot_products))
    angle_diff = torch.min(angle_diff, antipodal_angle_diff)

    losses = angle_diff ** 2

    return losses.mean() if reduce else losses




def quaternion_geodesic_loss(q_pred, q_true, epsilon=1e-7): #Redhwan for quaternions
    """
    Calculate geodesic loss between predicted and true quaternions.

    Parameters:
    - q_pred: Predicted quaternions (batch_size x 4)
    - q_true: True quaternions (batch_size x 4)
    - epsilon: Small value to avoid division by zero during normalization

    Returns:
    - geodesic_loss: Geodesic loss
    """

    # Ensure unit norm for quaternions
    q_pred = F.normalize(q_pred + epsilon, p=2, dim=-1)
    q_true = F.normalize(q_true + epsilon, p=2, dim=-1)

    # Calculate inner product and clip to [-1, 1] to avoid numerical issues
    dot_product = torch.clamp(torch.sum(q_pred * q_true, dim=-1), -1.0 + epsilon, 1.0 - epsilon)

    # Calculate geodesic distance
    angle = 2.0 * torch.acos(dot_product)

    # Check for NaN values in angle
    if torch.isnan(angle).any():
        raise ValueError("NaN values encountered in angle. Adjust normalization or input values.")

    # Calculate geodesic loss
    geodesic_loss = torch.mean(angle)

    return geodesic_loss


def quaternion_geodesic_loss_Red(q_pred, q_true, reduce=True):
    """
    Calculate geodesic loss between predicted and true quaternions.

    Parameters:
    - q_pred: Predicted quaternions (batch_size x 4)
    - q_true: True quaternions (batch_size x 4)
    - reduce: If True, compute the mean loss; if False, return per-sample losses.

    Returns:
    - geodesic_loss: Geodesic loss
    """

    # Ensure unit norm for quaternions
    q_pred = F.normalize(q_pred, p=2, dim=-1)
    q_true = F.normalize(q_true, p=2, dim=-1)
    # q_pred = F.normalize(q_pred + epsilon, p=2, dim=-1)
    # q_true = F.normalize(q_true + epsilon, p=2, dim=-1)

    # Calculate geodesic distance using quat_norm_diff
    angle = 2.0 * torch.asin(quat_norm_diff(q_pred, q_true) / 2.0) # asin

    if reduce:
        geodesic_loss = torch.mean(angle)
    else:
        geodesic_loss = angle

    return geodesic_loss


def quaternion_geodesic_loss_Red_epilon(q_pred, q_true, epsilon=1e-8, reduce=True):
    """
    Calculate geodesic loss between predicted and true quaternions.

    Parameters:
    - q_pred: Predicted quaternions (batch_size x 4)
    - q_true: True quaternions (batch_size x 4)
    - reduce: If True, compute the mean loss; if False, return per-sample losses.

    Returns:
    - geodesic_loss: Geodesic loss
    """

    # Ensure unit norm for quaternions
    # q_pred = F.normalize(q_pred, p=2, dim=-1)
    # q_true = F.normalize(q_true, p=2, dim=-1)
    q_pred = F.normalize(q_pred + epsilon, p=2, dim=-1)
    q_true = F.normalize(q_true + epsilon, p=2, dim=-1)

    # Calculate geodesic distance using quat_norm_diff
    angle = 2.0 * torch.asin(quat_norm_diff(q_pred, q_true) / 2.0)  # asin

    if reduce:
        geodesic_loss = torch.mean(angle)
    else:
        geodesic_loss = angle

    return geodesic_loss

def normalize_antipodal_quaternion(q):
    """
    Normalize quaternion to be antipodal, ensuring q and -q represent the same rotation.

    Parameters:
    - q: Quaternions (batch_size x 4)

    Returns:
    - normalized_q: Normalized antipodal quaternions
    """
    # Ensure unit norm for quaternions
    normalized_q = F.normalize(q, p=2, dim=-1)

    # Make sure the scalar part is positive
    normalized_q[torch.isnan(normalized_q)] = 0  # Handle division by zero
    normalized_q = torch.sign(normalized_q[:, 0]).view(-1, 1) * normalized_q

    return normalized_q

def quaternion_Anti_Geodesic_loss(q_pred, q_true):
    """
    Quaternion Geodesic Loss between predicted and true quaternions.

    Parameters:
    - q_pred: Predicted quaternions (batch_size x 4)
    - q_true: True quaternions (batch_size x 4)

    Returns:
    - geodesic_loss: Quaternion Geodesic Loss
    """
    # Normalize quaternions to be antipodal
    q_pred = normalize_antipodal_quaternion(q_pred)
    q_true = normalize_antipodal_quaternion(q_true)

    # Calculate dot product between quaternions
    dot_product = torch.sum(q_pred * q_true, dim=-1)

    # Clip dot product to be in the valid range [-1, 1]
    dot_product = torch.clamp(dot_product, -1.0, 1.0)

    # Calculate angular difference
    angular_diff = 2.0 * torch.acos(dot_product)

    # Convert angular difference to degrees
    angular_diff_degrees = angular_diff * (180.0 / 3.14159265358979323846)

    # Quaternion Geodesic Loss is half of the mean angular difference
    geodesic_loss = torch.mean(angular_diff_degrees) / 2.0

    return geodesic_loss

def quaternion_mse_loss(q_pred, q_true): #Redhwan
    """
    Calculate Mean Squared Error (MSE) loss between predicted and true quaternions.

    Parameters:
    - q_pred: Predicted quaternions (batch_size x 4)
    - q_true: True quaternions (batch_size x 4)

    Returns:
    - mse_loss: MSE loss
    """

    # Ensure unit norm for quaternions
    q_pred = F.normalize(q_pred, p=2, dim=-1)
    q_true = F.normalize(q_true, p=2, dim=-1)

    # Calculate MSE loss
    mse_loss = nn.MSELoss()(q_pred, q_true)

    return mse_loss

def quaternion_loss(q_pred, q_true): #Redhwan
    """
    Custom quaternion loss function.

    Parameters:
    - q_pred: Predicted quaternions (batch_size x 4)
    - q_true: True quaternions (batch_size x 4)

    Returns:
    - loss: Custom quaternion loss
    """

    # Ensure unit norm for quaternions
    q_pred = F.normalize(q_pred, p=2, dim=-1)
    q_true = F.normalize(q_true, p=2, dim=-1)

    # Calculate the quaternion difference
    q_diff = q_pred - q_true

    # Customize the loss based on your requirements
    # For example, you might want to penalize differences in certain quaternion components differently
    loss = torch.sum(q_diff**2)

    return loss

def unit_quaternion_regularization_loss(predicted_quat, target_quat): #Redhwan
    """
    Unit quaternion regularization loss.

    Args:
    - predicted_quat (torch.Tensor): Predicted quaternions (batch_size x 4).
    - target_quat (torch.Tensor): Target quaternions (batch_size x 4).

    Returns:
    - loss (torch.Tensor): Regularization loss.
    """
    # Normalize quaternions
    predicted_quat = predicted_quat / quaternion_norm(predicted_quat)
    target_quat = target_quat / quaternion_norm(target_quat)

    # Calculate the mean squared error between predicted and target quaternions
    mse_loss = nn.MSELoss()(predicted_quat, target_quat)

    return mse_loss


def quaternion_to_rotation_matrix(q): #Redhwan
    """
    Convert a quaternion to a 3x3 rotation matrix.
    Args:
        q (torch.Tensor): Input quaternion (batch_size x 4).
    Returns:
        torch.Tensor: Rotation matrices (batch_size x 3 x 3).
    """
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    N = q.shape[0]
    q2 = q * q
    R = torch.zeros(N, 3, 3, device=q.device, dtype=q.dtype)
    R[:, 0, 0] = 1 - 2 * (q2[:, 2] + q2[:, 3])
    R[:, 0, 1] = 2 * (q[:, 1] * q[:, 2] - q[:, 0] * q[:, 3])
    R[:, 0, 2] = 2 * (q[:, 0] * q[:, 2] + q[:, 1] * q[:, 3])
    R[:, 1, 0] = 2 * (q[:, 1] * q[:, 2] + q[:, 0] * q[:, 3])
    R[:, 1, 1] = 1 - 2 * (q2[:, 1] + q2[:, 3])
    R[:, 1, 2] = 2 * (q[:, 2] * q[:, 3] - q[:, 0] * q[:, 1])
    R[:, 2, 0] = 2 * (q[:, 1] * q[:, 3] - q[:, 0] * q[:, 2])
    R[:, 2, 1] = 2 * (q[:, 0] * q[:, 1] + q[:, 2] * q[:, 3])
    R[:, 2, 2] = 1 - 2 * (q2[:, 1] + q2[:, 2])
    return R

def lie_algebra_loss(y_true, y_pred): #Redhwan
    """
    Calculates the Lie algebra loss between two quaternion rotations.
    Args:
        y_true (torch.Tensor): True quaternion rotations (batch_size x 4).
        y_pred (torch.Tensor): Predicted quaternion rotations (batch_size x 4).
    Returns:
        torch.Tensor: Lie algebra loss.
    """
    R_true = quaternion_to_rotation_matrix(y_true)
    R_pred = quaternion_to_rotation_matrix(y_pred)

    # Calculate the difference between the rotation matrices
    diff_matrix = torch.bmm(R_true, R_pred.transpose(1, 2))
    trace = torch.einsum('bii->b', diff_matrix)

    # Calculate the Lie algebra loss
    loss = torch.mean(torch.acos((trace - 1) / 2))

    return loss

def quaternion_angular_error_loss(q_pred, q_true): #Redhwan
    """
    Quaternion Angular Error (QAE) loss between predicted and true quaternions.

    Parameters:
    - q_pred: Predicted quaternions (batch_size x 4)
    - q_true: True quaternions (batch_size x 4)

    Returns:
    - qae_loss: Quaternion Angular Error loss
    """

    # Ensure unit norm for quaternions
    q_pred = F.normalize(q_pred, p=2, dim=-1)
    q_true = F.normalize(q_true, p=2, dim=-1)

    # Calculate dot product between quaternions
    dot_product = torch.sum(q_pred * q_true, dim=-1)

    # Clip dot product to be in the valid range [-1, 1]
    dot_product = torch.clamp(dot_product, -1.0, 1.0)

    # Calculate angular difference
    angular_diff = 2.0 * torch.acos(dot_product)

    # Take the mean or sum of the angular differences to reduce to a scalar
    angular_diff = torch.mean(angular_diff)  # or torch.sum(angular_diff)

    # Convert angular difference to degrees
    angular_diff_degrees = angular_diff * (180.0 / 3.14159265358979323846)

    # QAE loss is half of the angular difference
    qae_loss = angular_diff_degrees / 2.0

    return qae_loss


def quaternion_angular_error_loss_eps(q_pred, q_true, epsilon=1e-7):
    """
    Quaternion Angular Error (QAE) loss between predicted and true quaternions.

    Parameters:
    - q_pred: Predicted quaternions (batch_size x 4)
    - q_true: True quaternions (batch_size x 4)

    Returns:
    - qae_loss: Quaternion Angular Error loss
    """

    # Ensure unit norm for quaternions
    q_pred = F.normalize(q_pred + epsilon, p=2, dim=-1)
    q_true = F.normalize(q_true + epsilon, p=2, dim=-1)

    # Calculate dot product between quaternions
    dot_product = torch.sum(q_pred * q_true, dim=-1)

    # Clip dot product to be in the valid range [-1, 1]
    dot_product = torch.clamp(dot_product, -1.0 + epsilon, 1.0 - epsilon)


    # Calculate angular difference
    angular_diff = 2.0 * torch.acos(dot_product)

    # Take the mean of the angular differences to reduce to a scalar
    angular_diff = torch.mean(angular_diff)

    # Convert angular difference to degrees
    # angular_diff_degrees = angular_diff * (180.0 / 3.14159265358979323846)
    angular_diff_degrees = angular_diff

    # QAE loss is half of the angular difference
    qae_loss = angular_diff_degrees / 2.0

    return qae_loss


def quaternion_angular_error_loss_eps_abs(q_pred, q_true, epsilon=1e-7):
    """
    Quaternion Angular Error (QAE) loss between predicted and true quaternions.

    Parameters:
    - q_pred: Predicted quaternions (batch_size x 4)
    - q_true: True quaternions (batch_size x 4)

    Returns:
    - qae_loss: Quaternion Angular Error loss
    """

    # Ensure unit norm for quaternions
    q_pred = F.normalize(q_pred + epsilon, p=2, dim=-1)
    q_true = F.normalize(q_true + epsilon, p=2, dim=-1)

    # Calculate dot product between quaternions
    dot_product = torch.sum(q_pred * q_true, dim=-1)

    # Clip dot product to be in the valid range [0, pi]
    dot_product = torch.clamp(dot_product, -1.0 + epsilon, 1.0 - epsilon)
    dot_product = abs(dot_product)

    # Calculate angular difference
    angular_diff = 2.0 * torch.acos(dot_product)

    # Take the mean of the angular differences to reduce to a scalar
    angular_diff = torch.mean(angular_diff)

    # Convert angular difference to degrees
    # angular_diff_degrees = angular_diff * (180.0 / 3.14159265358979323846)
    angular_diff_degrees = angular_diff

    # QAE loss is half of the angular difference
    qae_loss = angular_diff_degrees / 2.0

    return qae_loss


def quaternion_angular_error_loss_eps_min(q_pred, q_true, epsilon=1e-7):
    """
    Quaternion Angular Error (QAE) loss between predicted and true quaternions.

    Parameters:
    - q_pred: Predicted quaternions (batch_size x 4)
    - q_true: True quaternions (batch_size x 4)

    Returns:
    - qae_loss: Quaternion Angular Error loss
    """

    # Ensure unit norm for quaternions
    q_pred = F.normalize(q_pred + epsilon, p=2, dim=-1)
    q_true = F.normalize(q_true + epsilon, p=2, dim=-1)

    # Calculate dot product between quaternions
    dot_product = torch.sum(q_pred * q_true, dim=-1)

    # Clip dot product to be in the valid range [0, pi]
    dot_product = torch.clamp(dot_product, -1.0 + epsilon, 1.0 - epsilon)
    dot_product = abs(dot_product)

    # Calculate angular difference
    angular_diff = 2.0 * torch.acos(dot_product)
    angular_diff_complement = 2.0 * (3.14159265358979323846 - torch.acos(dot_product))
    min_angular_diff = torch.min(angular_diff, angular_diff_complement)

    # Take the mean of the angular differences to reduce to a scalar
    angular_diff = torch.mean(min_angular_diff)

    # Convert angular difference to degrees
    # angular_diff_degrees = angular_diff * (180.0 / 3.14159265358979323846)
    angular_diff_degrees = angular_diff

    # QAE loss is half of the angular difference
    qae_loss = angular_diff_degrees / 2.0

    return qae_loss


# def wrapped_loss(y_pred, y_true):
def wrapped_loss(q_pred, q_true, reduce=True):
    """
    Calculate geodesic loss between predicted and true quaternions.

    Parameters:
    - q_pred: Predicted quaternions (batch_size x 4)
    - q_true: True quaternions (batch_size x 4)
    - reduce: If True, compute the mean loss; if False, return per-sample losses.

    Returns:
    - geodesic_loss: Geodesic loss
    """

    # Ensure unit norm for quaternions
    q_pred = F.normalize(q_pred, p=2, dim=-1)
    q_true = F.normalize(q_true, p=2, dim=-1)

    # Calculate geodesic distance using quat_norm_diff
    diff = torch.atan2(torch.sin(quat_norm_diff(q_pred, q_true) / 2.0), torch.cos(quat_norm_diff(q_pred, q_true) / 2.0))
    squared_error = torch.pow(diff, 2)

    if reduce:
        wrapped_loss = torch.mean(squared_error)
    else:
        wrapped_loss = squared_error

    return wrapped_loss


def quaternion_distance_loss(q_pred, q_true): #Redhwan
    """
    Quaternion Distance Loss between predicted and true quaternions.

    Parameters:
    - q_pred: Predicted quaternions (batch_size x 4)
    - q_true: True quaternions (batch_size x 4)

    Returns:
    - distance_loss: Quaternion Distance Loss
    """

    # Ensure unit norm for quaternions
    q_pred = F.normalize(q_pred, p=2, dim=-1)
    q_true = F.normalize(q_true, p=2, dim=-1)

    # Compute the Euclidean distance between quaternions
    distance_loss = torch.norm(q_pred - q_true, p=2, dim=-1)

    # Take the mean or sum of the distances to reduce to a scalar
    distance_loss = torch.mean(distance_loss)  # or torch.sum(distance_loss)

    return distance_loss



# Angle-Axis Loss
def angle_axis_loss(predicted, ground_truth):
    predicted_norm = torch.norm(predicted)
    ground_truth_norm = torch.norm(ground_truth)
    dot_product = torch.sum(predicted * ground_truth)
    angle = torch.acos(dot_product / (predicted_norm * ground_truth_norm))
    loss = 1 - torch.cos(angle)
    return loss

#


def l2_ordinal_loss(predicted, ground_truth, num_classes=4, l2_weight=0.99): #0.9
    predicted = F.normalize(predicted, p=2, dim=-1)
    ground_truth = F.normalize(ground_truth, p=2, dim=-1)
    # L2 (Mean Squared Error) regression loss
    l2_loss = F.mse_loss(predicted, ground_truth)

    # Ordinal regression loss
    ordinal_targets = torch.arange(1, num_classes + 1, dtype=torch.float32, device=predicted.device)

    # Unsqueeze to match the batch size of predicted
    ordinal_targets = ordinal_targets.unsqueeze(0).expand_as(predicted)

    ordinal_loss = F.mse_loss(predicted, ordinal_targets)

    # Combine the losses
    combined_loss = l2_weight * l2_loss + (1 - l2_weight) * ordinal_loss

    return combined_loss