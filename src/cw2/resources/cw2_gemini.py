import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim


# --- 2. Carlini & Wagner Attack Implementation ---
class CarliniWagnerAttack(object):
    def __init__(self, model, device, num_classes=10, targeted=False, confidence=0,
                 learning_rate=0.01, binary_search_steps=9, max_iterations=1000,
                 abort_early=True, initial_const=0.01, norm='L2'):
        self.model = model
        self.num_classes = num_classes
        self.targeted = targeted
        self.confidence = confidence
        self.learning_rate = learning_rate
        self.binary_search_steps = binary_search_steps
        self.max_iterations = max_iterations
        self.abort_early = abort_early
        self.initial_const = initial_const
        self.norm = norm
        self.device = device

        # Define a clamp function to ensure pixel values stay in [0, 1]
        self.tanh_transform = lambda x: 0.5 * (torch.tanh(x) + 1)
        self.inverse_tanh_transform = lambda x: torch.atanh(x * 2 - 1)

    def _loss_function(self, outputs, labels):
        # outputs: logits from the model
        # labels: true labels for untargeted, target labels for targeted

        one_hot_labels = torch.zeros(outputs.size()).to(self.device)
        one_hot_labels.scatter_(1, labels.unsqueeze(1), 1)

        real = torch.sum(one_hot_labels * outputs, dim=1)
        # Set true class to a very small number
        other = torch.max((1 - one_hot_labels) * outputs -
                          one_hot_labels * 10000, dim=1)[0]

        if self.targeted:
            # If targeted, we want target class to be most likely
            # Maximize (other - real)
            loss1 = torch.clamp(other - real + self.confidence, min=0.)
        else:
            # If untargeted, we want true class to be least likely
            # Maximize (real - other)
            loss1 = torch.clamp(real - other + self.confidence, min=0.)

        return torch.sum(loss1)

    def attack(self, original_images, labels):
        batch_size = original_images.shape[0]

        # Convert original images to tanh space for optimization
        original_images_tanh = self.inverse_tanh_transform(
            original_images).detach()

        # Initialize the adversarial perturbation variable (w)
        # We optimize 'w' and then transform it to 'x_adv'
        w = torch.zeros_like(original_images_tanh,
                             requires_grad=True).to(self.device)
        optimizer = optim.Adam([w], lr=self.learning_rate)

        # Binary search for 'c'
        lower_bound = torch.zeros(batch_size).to(self.device)
        upper_bound = torch.ones(batch_size) * 1e10  # A large upper bound
        c = (torch.ones(batch_size) * self.initial_const).to(self.device)

        # Store the best adversarial examples found
        o_best_adv_images = original_images.clone().detach()
        o_best_L2 = torch.full((batch_size,), float('inf')).to(self.device)
        # Store the max logit of misclassified class for untargeted/target class for targeted
        o_best_score = -1 * torch.ones(batch_size).to(self.device)

        for binary_step in range(self.binary_search_steps):
            # Reset w for each binary search step (important for finding optimal c)
            w.data = original_images_tanh.clone()

            best_adv_images = original_images.clone().detach()
            best_L2 = torch.full((batch_size,), float('inf')).to(self.device)
            best_score = -1 * torch.ones(batch_size).to(self.device)

            for iteration in range(self.max_iterations):
                # Apply tanh transform to get current adversarial image
                x_adv = self.tanh_transform(w)

                # Ensure pixels stay within [0, 1] range after adding delta
                # This is implicitly handled by tanh, but explicit clipping might be needed
                # if you were not using tanh transform, or to be extra safe.
                # Here, the tanh transform ensures it.

                # Calculate L2 norm of the perturbation
                L2_norm = torch.sum(
                    (x_adv - original_images)**2, dim=[1, 2, 3]).float()

                # Get model outputs (logits)
                outputs = self.model(x_adv)

                # Calculate classification loss
                class_loss = self._loss_function(outputs, labels)

                # Total loss
                loss = torch.sum(L2_norm * c) + class_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Evaluate current adversarial examples
                with torch.no_grad():
                    preds = torch.argmax(outputs, dim=1)

                    # Check if misclassified (or classified as target)
                    if self.targeted:
                        # Success if predicted class is target
                        is_successful = (preds == labels)
                    else:
                        # Success if predicted class is NOT original class
                        # Assuming labels are original true labels
                        is_successful = (preds != labels)

                    # For debugging, print for the first example in the batch
                    if iteration % 50 == 0:  # Print every 50 iterations
                        print(
                            f"Binary Step {binary_step+1}/{self.binary_search_steps}, Iteration {iteration+1}/{self.max_iterations}")
                        print(f"Current c: {c[0].item():.4f}")
                        print(
                            f"Total Loss: {loss.item():.4f}, Class Loss: {class_loss.item():.4f}, L2 Norm: {L2_norm[0].item():.4f}")
                        print(
                            f"Original Label: {labels[0].item()}, Current Pred: {preds[0].item()}")
                        print(
                            f"Success (current iteration): {is_successful[0].item()}")

                    # Update best adversarial examples for the current c
                    for i in range(batch_size):
                        if is_successful[i]:
                            if L2_norm[i] < best_L2[i]:
                                best_L2[i] = L2_norm[i]
                                best_adv_images[i] = x_adv[i]
                                # Score of target class or misclassified class
                                best_score[i] = outputs[i, labels[i]
                                                        ] if self.targeted else outputs[i, preds[i]]

            # After max_iterations, update overall best for binary search
            for i in range(batch_size):
                if best_L2[i] < o_best_L2[i]:
                    o_best_L2[i] = best_L2[i]
                    o_best_adv_images[i] = best_adv_images[i]
                    o_best_score[i] = best_score[i]

            # Adjust c for the next binary search step
            for i in range(batch_size):
                if o_best_score[i] != -1:  # If an adversarial example was found
                    upper_bound[i] = c[i]
                    c[i] = (lower_bound[i] + upper_bound[i]) / 2
                else:
                    lower_bound[i] = c[i]
                    c[i] = (lower_bound[i] + upper_bound[i]) / 2
                    if upper_bound[i] < 1e9:  # To prevent c from growing indefinitely
                        c[i] *= 10  # Increase c if no success

        return o_best_adv_images
