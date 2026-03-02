use nalgebra::{Matrix3, Vector3};

/// Represents the geometric state of a 2-node sequence in the CIC Tokenizer
pub struct TokenizerNode {
    pub theta: Vector3<f64>, // [ P(X1=1), P(X2=1|X1=0), P(X2=1|X1=1) ]
    pub n_samples: f64,      // Empirical mass (frequency of token pair)
}

impl TokenizerNode {
    /// Computes the exact entropic repulsion vector analytically
    pub fn compute_entropy_gradient(&self) -> Vector3<f64> {
        let eps = 1e-12;
        Vector3::new(
            ((1.0 - self.theta[0] + eps) / (self.theta[0] + eps)).ln(),
            ((1.0 - self.theta[1] + eps) / (self.theta[1] + eps)).ln(),
            ((1.0 - self.theta[2] + eps) / (self.theta[2] + eps)).ln(),
        )
    }

    /// Analytically computes the inverse Fisher Information Matrix for the X1 -> X2 DAG.
    pub fn analytical_inverse_fisher(&self) -> Matrix3<f64> {
        let eps = 1e-12; 
        
        let t1 = self.theta[0];
        let t2_given_0 = self.theta[1];
        let t2_given_1 = self.theta[2];

        let g_inv_11 = t1 * (1.0 - t1);
        let g_inv_22 = (t2_given_0 * (1.0 - t2_given_0)) / (1.0 - t1 + eps);
        let g_inv_33 = (t2_given_1 * (1.0 - t2_given_1)) / (t1 + eps);

        Matrix3::new(
            g_inv_11, 0.0,      0.0,
            0.0,      g_inv_22, 0.0,
            0.0,      0.0,      g_inv_33,
        )
    }

    /// Executes one step of the Curvature-Regularized Ricci Flow
    pub fn geometric_step(&mut self, score_u: Vector3<f64>, grad_r: Vector3<f64>, kappa: f64) {
        let grad_h = self.compute_entropy_gradient();

        let total_force = score_u 
            + grad_h.scale(1.0 / self.n_samples) 
            - grad_r.scale(1.0 / self.n_samples);

        let g_inv = self.analytical_inverse_fisher(); 
        let natural_step = g_inv * total_force;

        self.theta = (self.theta + natural_step.scale(kappa))
            .map(|x| x.clamp(0.01, 0.99));
    }
}
