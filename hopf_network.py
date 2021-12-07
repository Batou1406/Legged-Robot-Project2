"""
CPG in polar coordinates based on:
Pattern generators with sensory feedback for the control of quadruped
authors: L. Righetti, A. Ijspeert
https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=4543306

"""
import time
import numpy as np
import matplotlib
from sys import platform
if platform =="darwin": # mac
  import PyQt5
  matplotlib.use("Qt5Agg")
else: # linux
  matplotlib.use('TkAgg')

from matplotlib import pyplot as plt
from env.quadruped_gym_env import QuadrupedGymEnv


class HopfNetwork():
  """ CPG network based on hopf polar equations mapped to foot positions in Cartesian space.

  Foot Order is FR, FL, RR, RL
  (Front Right, Front Left, Rear Right, Rear Left)
  """
  def __init__(self,
                mu=1**2,                # converge to sqrt(mu)
                omega_swing=1*2*np.pi,  # MUST EDIT
                omega_stance=1*2*np.pi, # MUST EDIT
                gait="WALK",            # change depending on desired gait
                coupling_strength=1,    # coefficient to multiply coupling matrix
                couple=True,            # should couple
                time_step=0.001,        # time step
                ground_clearance=0.05,  # foot swing height
                ground_penetration=0.01,# foot stance penetration into ground
                robot_height=0.25,      # in nominal case (standing)
                des_step_len=0.04,      # desired step length
                ):

    ###############
    # initialize CPG data structures: amplitude is row 0, and phase is row 1
    self.X = np.zeros((2,4))

    # save parameters
    self._mu = mu
    self._omega_swing = omega_swing
    self._omega_stance = omega_stance
    self._couple = couple
    self._coupling_strength = coupling_strength
    self._dt = time_step
    self._set_gait(gait)

    # set oscillator initial conditions
    self.X[0,:] = np.random.rand(4) * .1
    self.X[1,:] = self.PHI[0,:]

    # save body and foot shaping
    self._ground_clearance = ground_clearance
    self._ground_penetration = ground_penetration
    self._robot_height = robot_height
    self._des_step_len = des_step_len


  def _set_gait(self,gait):
    """ For coupling oscillators in phase space.
    [TODO] update all coupling matrices
    """
    self.PHI_trot = np.array([[0,np.pi,np.pi,0],
                              [np.pi,0,0,np.pi],
                              [np.pi,0,0,np.pi],
                              [0,np.pi,np.pi,0]])
    self.PHI_walk = np.array([[0,np.pi,3*np.pi/2,np.pi/2],
                              [np.pi,0,np.pi/2,3*np.pi/2],
                              [np.pi/2,3*np.pi/2,0,np.pi],
                              [3*np.pi/2,np.pi/2,np.pi,0]])
    self.PHI_bound = np.array([[0,0,np.pi,np.pi],
                               [0,0,np.pi,np.pi],
                               [np.pi,np.pi,0,0],
                               [np.pi,np.pi,0,0]])
    self.PHI_pace = np.array([[0,np.pi,0,np.pi],
                              [np.pi,0,np.pi,0],
                              [0,np.pi,0,np.pi],
                              [np.pi,0,np.pi,0]])

    if gait == "TROT":
      print('TROT')
      self.PHI = self.PHI_trot
      self._omega_swing = 2*np.pi*7
      self._omega_stance = 3*np.pi*7
    elif gait == "PACE":
      print('PACE')
      self.PHI = self.PHI_pace
      self._omega_swing = 2*np.pi*7
      self._omega_stance = 3*np.pi*7
    elif gait == "BOUND":
      print('BOUND')
      self.PHI = self.PHI_bound
      self._omega_swing = 2*np.pi*8
      self._omega_stance = 3*np.pi*8
    elif gait == "WALK":
      print('WALK')
      self._omega_swing = -2*np.pi*8
      self._omega_stance = -3*np.pi*8
      self.PHI = self.PHI_walk
    else:
      raise ValueError( gait + 'not implemented.')


  def update(self):
    """ Update oscillator states. """

    # update parameters, integrate
    self._integrate_hopf_equations()

    # map CPG variables to Cartesian foot xz positions (Equations 8, 9)
    #faut-il déclarer la taille de z ? oui
    z = np.zeros(4)
    x = -self._des_step_len*self.X[0,:]*np.cos(self.X[1,:]) # [TODO]
    for j in range(4):
        if(np.sin(self.X[1][j]>0)):
            z[j] = -self._robot_height + self._ground_clearance*np.sin(self.X[1][j]) # [TODO]
        else:
            z[j] = -self._robot_height + self._ground_penetration*np.sin(self.X[1][j]) # [TODO]
    return x, z


  def _integrate_hopf_equations(self):
    """ Hopf polar equations and integration. Use equations 6 and 7. """
    # bookkeeping - save copies of current CPG states
    X = self.X.copy()
    X_dot = np.zeros((2,4))
    alpha = 50
    r=np.zeros(4)
    theta=np.zeros(4)
    r_dot=np.zeros(4)
    theta_dot=np.zeros(4)
    # loop through each leg's oscillator
    for i in range(4):
      # get r_i, theta_i from X
      r[i], theta[i] = X[0][i], X[1][i] # [TODO]
      # compute r_dot (Equation 6)
      r_dot[i] = alpha*(self._mu - r[i]**2)*r[i]# [TODO]
      # determine whether oscillator i is in swing or stance phase to set natural frequency omega_swing or omega_stance (see Section 3)
      if theta[i] <= np.pi:
        theta_dot[i] = self._omega_swing # [TODO]
      else:
        theta_dot[i] = self._omega_stance

      # loop through other oscillators to add coupling (Equation 7)
      if self._couple:
          for j in range(4):
              theta_dot[i]+=X[0][j]*self._coupling_strength*np.sin(X[1][j]-X[1][i]-self.PHI[i][j])
          #theta_dot[i] += sum(X[0][j]*self._coupling_strength*np.sin(X[1][j]-X[1][i]-self.PHI[i][j]) for j in range[4]) # [TODO]

      # set X_dot[:,i]
      X_dot[:,i] = [r_dot[i], theta_dot[i]]

    # integrate
    self.X += X_dot*self._dt  # [TODO]
    # mod phase variables to keep between 0 and 2pi
    self.X[1,:] = self.X[1,:] % (2*np.pi)



if __name__ == "__main__":

  ADD_CARTESIAN_PD = True
  TIME_STEP = 0.001
  foot_y = 0.0838 # this is the hip length
  sideSign = np.array([-1, 1, -1, 1]) # get correct hip sign (body right is negative)

  env = QuadrupedGymEnv(render=True,              # visualize
                      on_rack=False,              # useful for debugging!
                      isRLGymInterface=False,     # not using RL
                      time_step=TIME_STEP,
                      action_repeat=1,
                      motor_control_mode="TORQUE",
                      add_noise=False,    # start in ideal conditions
                      # record_video=True
                      )

  # initialize Hopf Network, supply gait
  cpg = HopfNetwork(time_step=TIME_STEP)

  TEST_STEPS = int(10 / (TIME_STEP))
  t = np.arange(TEST_STEPS)*TIME_STEP

  # [TODO] initialize data structures to save CPG and robot states
  amp_deph=np.zeros((2,4,TEST_STEPS))
  robot_states=np.zeros((12,TEST_STEPS))


  ############## Sample Gains
  # joint PD gains
  kp=np.array([150,70,70])
  kd=np.array([2,0.5,0.5])
  # Cartesian PD gains
  kpCartesian = np.diag([2500]*3)
  kdCartesian = np.diag([40]*3)


  for j in range(TEST_STEPS):
    # initialize torque array to send to motors
    action = np.zeros(12)
    # get desired foot positions from CPG
    xs,zs = cpg.update()
    # [TODO] get current motor angles and velocities for joint PD, see GetMotorAngles(), GetMotorVelocities() in quadruped.py
    q=env.robot.GetMotorAngles()
    dq=env.robot.GetMotorVelocities()

    # loop through desired foot positions and calculate torques
    for i in range(4):
      # initialize torques for legi
      tau = np.zeros(3)
      # get desired foot i pos (xi, yi, zi) in leg frame
      leg_xyz = np.array([xs[i],sideSign[i] * foot_y,zs[i]])
      # call inverse kinematics to get corresponding joint angles (see ComputeInverseKinematics() in quadruped.py)
      leq_q=env.robot.ComputeInverseKinematics(i,leg_xyz) # [TODO]
      # Add joint PD contribution to tau for leg i (Equation 4)
      tau+=kp*(leq_q-q[3*i:3*(i+1)]) + kd*(np.zeros(3)-dq[3*i:3*(i+1)])# [TODO]

      # add Cartesian PD contribution
      if ADD_CARTESIAN_PD:
        # Get current Jacobian and foot position in leg frame (see ComputeJacobianAndPosition() in quadruped.py)
        # [TODO]
        J, pos=env.robot.ComputeJacobianAndPosition(i)
        # Get current foot velocity in leg frame (Equation 2)
        # [TODO]
        v=J@dq[3*i:3*(i+1)] #mult bizarre parce que c'est pas que des numpy
        # Calculate torque contribution from Cartesian PD (Equation 5) [Make sure you are using matrix multiplications]
        tau+=np.transpose(J)@(kpCartesian@(leg_xyz-pos) + kdCartesian@(np.zeros(3)-v)) # [TODO]
      # Set tau for legi in action vector
      action[3*i:3*i+3] = tau

    # send torques to robot and simulate TIME_STEP seconds
    env.step(action)

    # [TODO] save any CPG or robot states
    amp_deph[:,:,j]=cpg.X
    robot_states[:,j]=q



  #####################################################
  # PLOTS
  #####################################################
  # example
  # fig = plt.figure()
  # plt.plot(t,joint_pos[1,:], label='FR thigh')
  # plt.legend()
  # plt.show()
