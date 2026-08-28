import numpy as np

from potion.optimization.gradient_descent import Adam, RMS


def test_adam_accumulates_moments():
    optimizer = Adam(alpha=0.1, beta_1=0.5, beta_2=0.5, eps=0.)

    first_step = optimizer(np.array([1., 2.]))
    second_step = optimizer(np.array([3., 4.]))

    assert np.allclose(first_step, [0.1, 0.1])
    expected_m = np.array([7. / 3., 10. / 3.])
    expected_v = np.array([19. / 3., 12.])
    assert np.allclose(second_step, 0.1 * expected_m / np.sqrt(expected_v))


def test_adam_reset_discards_accumulated_moments():
    optimizer = Adam(alpha=0.1, beta_1=0.5, beta_2=0.5, eps=0.)
    gradient = np.array([3., 4.])
    optimizer(np.array([1., 2.]))

    reset_step = optimizer(gradient, reset=True)
    fresh_step = Adam(alpha=0.1, beta_1=0.5, beta_2=0.5, eps=0.)(
        gradient
    )

    assert np.allclose(reset_step, fresh_step)


def test_adam_returns_one_update_per_parameter():
    optimizer = Adam()

    update = optimizer(np.ones(3))

    assert update.shape == (3,)


def test_ema_rms_with_zero_beta_uses_squared_average_gradient():
    optimizer = RMS(alpha=0.1, beta=0., eps=0.)
    gradient = np.array([2., 3.])

    update = optimizer(gradient)

    expected_second_moment = gradient**2
    assert np.allclose(
        update,
        0.1 * gradient / np.sqrt(expected_second_moment),
    )


def test_ema_rms_with_zero_beta_does_not_retain_previous_gradient():
    optimizer = RMS(alpha=0.1, beta=0., eps=0.)
    gradient = np.array([2., 3.])

    optimizer(np.array([4., 5.]))
    update = optimizer(gradient)
    fresh_update = RMS(alpha=0.1, beta=0., eps=0.)(gradient)

    assert np.allclose(update, fresh_update)


def test_ema_rms_accumulates_second_moment():
    optimizer = RMS(alpha=0.1, beta=0.5, eps=0.)

    first_step = optimizer(np.array([1., 2.]))
    second_step = optimizer(np.array([3., 4.]))

    expected_first_q = np.array([0.5, 2.])
    expected_second_q = np.array([4.75, 9.])
    assert np.allclose(
        first_step, 0.1 * np.array([1., 2.]) / np.sqrt(expected_first_q)
    )
    assert np.allclose(
        second_step, 0.1 * np.array([3., 4.]) / np.sqrt(expected_second_q)
    )


def test_ema_rms_reset_discards_accumulated_second_moment():
    optimizer = RMS(alpha=0.1, beta=0.5, eps=0.)
    gradient = np.array([3., 4.])
    optimizer(np.array([1., 2.]))

    reset_step = optimizer(gradient, reset=True)
    fresh_step = RMS(alpha=0.1, beta=0.5, eps=0.)(gradient)

    assert np.allclose(reset_step, fresh_step)


def test_ema_rms_returns_one_update_per_parameter():
    optimizer = RMS()

    update = optimizer(np.ones(3))

    assert update.shape == (3,)
