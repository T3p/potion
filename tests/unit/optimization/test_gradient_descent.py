import numpy as np

from potion.optimization.gradient_descent import Adam


def test_adam_accumulates_moments():
    optimizer = Adam(alpha=0.1, beta_1=0.5, beta_2=0.5, eps=0.)

    first_step = optimizer(np.array([1., 2.]))
    second_step = optimizer(np.array([3., 4.]))

    assert np.allclose(first_step, [0.1, 0.1])
    expected_m = np.array([7. / 3., 10. / 3.])
    expected_v = np.array([19. / 3., 12.])
    assert np.allclose(second_step, 0.1 * expected_m / np.sqrt(expected_v))
