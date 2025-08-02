import torch
import numpy as np

if __name__ == '__main__':
    print('Built with CUDA:', torch.version.cuda)  # e.g. '11.7' or None
    print(
        'CUDA available:', torch.cuda.is_available()
    )  # False if no GPU support
    print('CUDA backends built:', torch.backends.cuda.is_built())  # False here

    print(np.random.randint(0, 1000))
    #######################################################################################

    # def edge_loss(
    #     self,
    #     edge: tuple[int, int],
    #     S_t: torch.Tensor = None,
    #     beta: float = None,
    #     penalized: bool = False,
    #     lambda_: float = None,
    #     F_ij: torch.Tensor = None,
    #     F_ji: torch.Tensor = None,
    #     S_i: torch.Tensor = None,
    #     S_j: torch.Tensor = None,
    #     D: torch.Tensor = None,
    #     gamma1: float = None,
    #     gamma2: float = None,
    # ) -> tuple[float, float, float]:
    #     """Compute the edge loss between two agents."""
    #     i, j = edge
    #     F_ij = self.agents[i].restriction_maps[j] if F_ij is None else F_ij
    #     F_ji = self.agents[j].restriction_maps[i] if F_ji is None else F_ji
    #     S_i = self.agents[i].sparse_representations[j] if S_i is None else S_i
    #     S_j = self.agents[j].sparse_representations[i] if S_j is None else S_i
    #     X_i = self.agents[i].X_train
    #     X_j = self.agents[j].X_train
    #     semantic_alignment = (
    #         torch.norm(F_ij @ X_i - F_ji @ X_j, p='fro')
    #         if S_t is None
    #         else torch.norm(S_t @ (F_ij @ X_i - F_ji @ X_j), p='fro')
    #     )
    #     communication_loss = (
    #         torch.norm(X_i - F_ij.T @ F_ji @ X_j, p='fro')
    #         + torch.norm(X_j - F_ji.T @ F_ij @ X_i, p='fro')
    #         if S_t is None
    #         else torch.norm(X_i - F_ij.T @ S_t @ F_ji @ X_j, p='fro')
    #         + torch.norm(X_j - F_ji.T @ S_t @ F_ij @ X_i, p='fro')
    #     )
    #     loss = (
    #         semantic_alignment
    #         if self.is_connection_graph
    #         else (1 - beta) * semantic_alignment + beta * communication_loss
    #     )

    #     assert (penalized == True) + (S_t is not None) != 1, (
    #         'penalized must be True if S_t is not None, and vice versa.'
    #     )
    #     if penalized:
    #         loss += lambda_ * torch.norm(torch.diag(S_t), p=1)

    #     if S_i is not None:
    #         loss += (
    #             torch.norm(X_i - D @ S_i, p=2)
    #             + torch.norm(X_j - D @ S_j, p=2)
    #             + gamma1 * torch.linalg.norm(S_j, ord=2, dim=1).sum()
    #             + gamma2 * torch.linalg.norm(S_i, ord=2, dim=1).sum()
    #         )
    #     return loss, semantic_alignment, communication_loss

    # def edge_augmented_lagrangian(
    #     self,
    #     edge: tuple[int, int],
    #     S_t: np.ndarray,
    #     alpha: float,
    #     U_i: np.ndarray,
    #     U_j: np.ndarray,
    #     Y_i: np.ndarray,
    #     Y_j: np.ndarray,
    #     beta: float = None,
    #     penalized: bool = None,
    #     lambda_: float = None,
    #     F_ij: torch.Tensor = None,
    #     F_ji: torch.Tensor = None,
    # ) -> float:
    #     U_i = torch.from_numpy(U_i.astype(np.float32))
    #     U_j = torch.from_numpy(U_j.astype(np.float32))
    #     Y_i = torch.from_numpy(Y_i.astype(np.float32))
    #     Y_j = torch.from_numpy(Y_j.astype(np.float32))
    #     S_t = (
    #         torch.from_numpy(S_t.astype(np.float32))
    #         if S_t is not None
    #         else None
    #     )
    #     F_ij = (
    #         torch.from_numpy(F_ij.astype(np.float32))
    #         if F_ij is not None
    #         else F_ij
    #     )
    #     F_ji = (
    #         torch.from_numpy(F_ji.astype(np.float32))
    #         if F_ji is not None
    #         else F_ji
    #     )
    #     loss, sem, comm = self.edge_loss(
    #         edge=edge,
    #         S_t=S_t,
    #         beta=beta,
    #         penalized=penalized,
    #         lambda_=lambda_,
    #         F_ij=F_ij,
    #         F_ji=F_ji,
    #     )
    #     dual_loss = (alpha / 2) * (
    #         torch.norm(F_ji.T - Y_j + U_j, p='fro') ** 2
    #         + torch.norm(F_ij.T - Y_i + U_i, p='fro') ** 2
    #     )
    #     return loss, sem, comm, dual_loss

    # def edge_loss_grad(
    #     self,
    #     edge: tuple[int, int],
    #     S_t: np.ndarray,
    #     beta: float,
    #     autodiff: bool = True,
    # ) -> np.ndarray:
    #     """Compute the edge loss gradient."""
    #     if autodiff:
    #         S_t = torch.tensor(
    #             S_t,
    #             dtype=torch.float32,
    #             device=self.device,
    #             requires_grad=True,
    #         )
    #         loss, _, _ = self.edge_loss(
    #             edge=edge,
    #             S_t=S_t,
    #             beta=beta,
    #             penalized=False,
    #         )
    #         loss.backward()
    #         # return S_t.grad.detach().cpu().numpy()
    #         return S_t.grad.numpy()
    #     else:
    #         return None

    # def compute_edge_losses(
    #     self,
    #     dictionary: torch.Tensor = None,
    #     beta: float = None,
    #     lambda_: float = None,
    #     gamma1: float = None,
    #     gamma2: float = None,
    # ) -> dict[tuple[int, int], float]:
    #     """Compute the edge losses for all edges in the graph."""
    #     edge_losses: dict[tuple[int, int], float] = {}
    #     penalized = False if lambda_ is None else True
    #     for i, j in self.graph.get_edgelist():
    #         edge_losses[(i, j)], _, _ = self.edge_loss(
    #             edge=(i, j),
    #             S_t=self.edge_masks[(i, j)],
    #             beta=beta,
    #             penalized=penalized,
    #             lambda_=lambda_,
    #             gamma1=gamma1,
    #             gamma2=gamma2,
    #             D=dictionary,
    #         )
    #     return edge_losses

    ###############################

    # def get_local_vars(
    #     self,
    #     varname: str,
    #     agent_idx: int,
    # ) -> np.ndarray:
    #     if varname == 'S':
    #         out = self.S[
    #             :,
    #             agent_idx * self.n_examples : (agent_idx + 1)
    #             * self.n_examples,
    #         ]
    #     elif varname == 'X':
    #         out = self.X[
    #             :,
    #             agent_idx * self.n_examples : (agent_idx + 1)
    #             * self.n_examples,
    #         ]
    #     elif varname == 'Z':
    #         out = self.Z[
    #             :,
    #             agent_idx * self.n_examples : (agent_idx + 1)
    #             * self.n_examples,
    #         ]
    #     elif varname == 'U':
    #         out = self.U[
    #             :,
    #             agent_idx * self.n_examples : (agent_idx + 1)
    #             * self.n_examples,
    #         ]
    #     else:
    #         raise ValueError(f"Unknown variable name '{varname}'.")
    #     return out

    # def assign_local_vars(
    #     self,
    #     value: np.ndarray,
    #     varname: str,
    #     agent_idx: int,
    # ) -> None:
    #     if varname == 'S':
    #         self.S[
    #             :,
    #             agent_idx * self.n_examples : (agent_idx + 1)
    #             * self.n_examples,
    #         ] = value
    #     elif varname == 'X':
    #         self.X[
    #             :,
    #             agent_idx * self.n_examples : (agent_idx + 1)
    #             * self.n_examples,
    #         ] = value
    #     elif varname == 'Z':
    #         self.Z[
    #             :,
    #             agent_idx * self.n_examples : (agent_idx + 1)
    #             * self.n_examples,
    #         ] = value
    #     elif varname == 'U':
    #         self.U[
    #             :,
    #             agent_idx * self.n_examples : (agent_idx + 1)
    #             * self.n_examples,
    #         ] = value
    #     else:
    #         raise ValueError(f"Unknown variable name '{varname}'.")
    #     return None
