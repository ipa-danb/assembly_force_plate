import rust_subscriber
import numpy as np

a = np.array([0.0, 0.0, 0.0])

rust_subscriber.start_subscriber(a)
