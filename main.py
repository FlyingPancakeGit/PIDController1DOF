import matplotlib.pyplot as plt
import numpy as np
import time

# setup
GRAVITY = -10


class Controller:
    def __init__(self, saturation_limit, k_p, k_i, k_d):
        self.k_p = k_p
        self.k_i = k_i
        self.k_d = k_d

        self.integral = 0
        self.derivative = 0
        self.previous_error = 0
        self.previous_error_time = 0

        self.saturation_limit = saturation_limit
        self.unclamped_signal = 0
        self.clamped_signal = 0

        # flags
        self.integral_on = True

    def check_saturated(self, error, unclamped_signal, clamped_signal):
        self.integral_on = not((unclamped_signal != clamped_signal) and (error*unclamped_signal>0))

    def calculate_actuating_signal(self, error, dt):
        if self.integral_on:
            self.integral += error

        if dt:
            self.derivative = (error - self.previous_error) / dt

        self.unclamped_signal = (error * self.k_p) + (self.integral * self.k_i) + (self.derivative * self.k_d)

        if self.saturation_limit:
            self.clamped_signal = max(0, min(self.unclamped_signal, self.saturation_limit))
            self.check_saturated(error, self.unclamped_signal, self.clamped_signal)

        return self.unclamped_signal


class Drone:
    def __init__(self, initial_pos, target_pos):
        # properties
        self.mass = 100
        self.max_thrust = 10000

        # state
        self.pos = initial_pos
        self.acceleration = 0
        self.velocity = 0
        self.thrust = 0

        # error correction
        self.velocity_controller = Controller(None, 5, 0, 0)
        self.acceleration_controller = Controller(self.max_thrust, 5, 0.1, 0.1)
        self.target_pos = target_pos
        self.error = 0

    def calculate_error(self):
        self.error = self.target_pos - self.pos

    def get_thrust(self, dt):
        self.target_velocity = self.velocity_controller.calculate_actuating_signal(self.error, dt)
        self.thrust = self.acceleration_controller.calculate_actuating_signal(self.target_velocity - self.velocity, dt)

    def update_motion(self, dt):
        self.acceleration = (self.thrust/self.mass) + GRAVITY
        self.velocity += self.acceleration * dt
        self.pos += self.velocity * dt

    def update(self, dt):
        self.calculate_error()
        self.get_thrust(dt)
        self.update_motion(dt)


class Main:
    def __init__(self):
        self.running = True
        self.drone = Drone(0, 100)

        # timings
        self.start_time = time.time()
        self.last_loop_time = 0
        self.last_log_time = 0

        # data
        self.data_x = np.empty(10000, dtype=float)
        self.data_y = np.empty( 10000, dtype=float)

        self.data_current_index = 0

    def log(self):
        if time.time() - self.last_log_time > 0.01:
            self.data_x[self.data_current_index] = time.time() - self.start_time
            self.data_y[self.data_current_index]= self.drone.pos
            self.data_current_index += 1
            self.last_log_time = time.time()

    def plot(self):
        plt.plot(self.data_x[:self.data_current_index], self.data_y[:self.data_current_index])
        plt.xlabel('Time (s)')
        plt.ylabel('Height (m)')
        plt.show()

    def main_loop(self):
        self.start_time = time.time()
        self.last_loop_time = self.start_time

        while self.running:
            dt = time.time() - self.last_loop_time
            self.last_loop_time = time.time()
            self.drone.update(dt)
            self.log()

            if self.data_current_index % 100 == 0:
                self.plot()


if __name__ == "__main__":
    main = Main()
    main.main_loop()
