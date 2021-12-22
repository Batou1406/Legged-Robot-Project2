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
                gait="TROT",            # change depending on desired gait
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
      self._omega_swing = 2*np.pi*6
      self._omega_stance = 3*np.pi*6
    elif gait == "PACE":
      print('PACE')
      self.PHI = self.PHI_pace
      self._omega_swing = 2*np.pi*7
      self._omega_stance = 3*np.pi*7
    elif gait == "BOUND":
      print('BOUND')
      self.PHI = self.PHI_bound
      self._omega_swing = 2*np.pi*11
      self._omega_stance = 3*np.pi*11
    elif gait == "WALK":
      print('WALK')
      self._omega_swing = 2*np.pi*8
      self._omega_stance = 3*np.pi*8
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

  env = QuadrupedGymEnv(render=False,              # visualize
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
  f_pos_x_fr = list("")
  f_pos_z_fr =  list("")
  f_pos_x_fl = list("")
  f_pos_z_fl =  list("")
  f_pos_x_rr = list("")
  f_pos_z_rr =  list("")
  f_pos_x_rl = list("")
  f_pos_z_rl =  list("")
  f_pos_z_compa = list("")
  velocity = list("")
  
  q_tot_zero = list("")
  q_tot_one = list("")
  q_tot_two = list("")
  qdes_tot_zero = list("")
  qdes_tot_one = list("")
  qdes_tot_two = list("")
  dq_tot = list("")
  desired_q = list("")
  desired_dq = list("")
  tautau = list("")
  tautau1 = list("")
  tautau2 = list("")
  tautau3 = list("")
  tau_der = list("")
  
  pos_x= list("")
  pos_y = list("")
  pos_z = list("")
  leg_x = list("")
  leg_y = list("")
  leg_z = list("")
  leg_x1 = list("")
  leg_y1 = list("")
  leg_z1 = list("")
  pos_z_2 = list("")
  
  speed = list("")
  desired_speed = list("")
  
  contacts = list("")


  ############## Sample Gains
  # joint PD gains
  kp=np.array([150,70,70])
  kd=np.array([2,0.5,0.5])
  # Cartesian PD gains
  kpCartesian = np.diag([3000]*3)
  kdCartesian = np.diag([80]*3)
  print(kpCartesian)


  for j in range(TEST_STEPS):
    # initialize torque array to send to motors
    action = np.zeros(12)
    # get desired foot positions from CPG
    xs,zs = cpg.update()
    # [TODO] get current motor angles and velocities for joint PD, see GetMotorAngles(), GetMotorVelocities() in quadruped.py
    q=env.robot.GetMotorAngles()
    dq=env.robot.GetMotorVelocities()
    vel = env.robot.GetBaseLinearVelocity()
   
    f_pos_x_fr.append(xs[0])
    f_pos_z_fr.append(zs[0])
    f_pos_x_fl.append(xs[1])
    f_pos_z_fl.append(zs[1])
    f_pos_x_rr.append(xs[2])
    f_pos_z_rr.append(zs[2])
    f_pos_x_rl.append(xs[3])
    f_pos_z_rl.append(zs[3])
    f_pos_z_compa.append(zs[2])

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

        if i ==0:
            pos_x.append(pos[0])
            pos_y.append(pos[1])
            pos_z.append(pos[2])
            leg_x.append(leg_xyz[0])
            leg_y.append(leg_xyz[1])
            leg_z.append(leg_xyz[2])
            qdes_tot_zero.append(leq_q[0])
            q_tot_zero.append(q[0])
            qdes_tot_one.append(leq_q[1])
            q_tot_one.append(q[1]) 
            qdes_tot_two.append(leq_q[2])
            q_tot_two.append(q[2]) 

      # Set tau for legi in action vector
      action[3*i:3*i+3] = tau
      velocity.append(vel[0])
    # send torques to robot and simulate TIME_STEP seconds
    env.step(action)

    # [TODO] save any CPG or robot states
    amp_deph[:,:,j]=cpg.X
    robot_states[:,j]=q



  #####################################################
  # PLOTS
  #####################################################
  # example
  fig1 =  plt.figure()
  plt.plot(t[0:10000],velocity[0:10000], label='x velocity')
  #plt.plot(t[0:1000],q_tot_zero[0:1000], label='joint angle leg 0')
  #plt.plot(t[0:1000],f_pos_x_rr[0:1000],'r', label = 'x leg2')
  #plt.plot(t[0:1000],f_pos_x_fl[0:1000], 'g',label='x leg 1')
  #plt.plot(t[0:1000],f_pos_x_rl[0:1000],'y', label = 'x leg 3')
  plt.legend()
  
  #fig2 = plt.figure()
  #plt.plot(t[0:1000],pos_z[0:1000], label='z leg 0')
  #plt.plot(t[0:1000],leg_z[0:1000], label='desired z leg 0')
  ##plt.plot(t[0:1000],f_pos_z_fl[0:1000], 'g',label='z leg 1')
  ##plt.plot(t[0:1000],f_pos_z_rl[0:1000],'y', label = 'z leg 3')
  #plt.legend()
  
  #fig3 = plt.figure()
  #plt.plot(t[0:1000],pos_y[0:1000], label='y leg 0')
  #plt.plot(t[0:1000],leg_y[0:1000], label='desired y leg 0')
  #plt.legend()
  ## plt.show()
  lim=1000
  fig, axs = plt.subplots(1, 3)
  fig.suptitle('Actual vs desired joints front right foot position without cartesian pd')

  axs[0].set_title('hip joint')
 
  axs[0].plot(t[0:lim],qdes_tot_zero[0:1000][0:lim])
  axs[0].plot(t[0:1000],q_tot_zero[0:lim])
  

  axs[1].set_title('thigh joint')
  axs[1].plot(t[0:lim],qdes_tot_one[0:1000][0:lim])
  axs[1].plot(t[0:1000],q_tot_one[0:lim])

  axs[2].set_title('calf joint')
  axs[2].plot(t[0:lim],qdes_tot_two[0:1000][0:lim])
  axs[2].plot(t[0:1000],q_tot_two[0:lim])

  plt.show()
