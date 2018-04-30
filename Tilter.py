import pymunk
from pymunk import Vec2d as v2 
import numpy as np 
import arcade 

class Ball(): 

	def __init__(self,lim, start_height = 380, size = 10): 

		self.lim = lim
		self.start_height = start_height
		self.size = size 

		pos = v2(np.random.randint(lim[0], lim[1]), start_height)
		# pos = v2(400,400)
		moment = pymunk.moment_for_circle(5, 0, self.size)
		self.body = pymunk.Body(self.size, moment)
		self.body.position = pos
		self.shape = pymunk.Circle(self.body, self.size)


class JointBar(): 

	def __init__(self, length = 200):

		self.joint_center = pymunk.Body(body_type = pymunk.Body.STATIC)
		self.joint_center.position =  v2(350,350)

		self.length = length
		self.body = pymunk.Body(100000, 100000)
		self.body.position = v2(350,350)
		self.shape = pymunk.Segment(self.body, (-self.length/2,0),(self.length/2,0), 5)

		self.constraint = pymunk.PinJoint(self.body, self.joint_center)


class World(): 

	def __init__(self, max_steps = 300, world_size = 700, x_lim = [250,450], continuous = False): 

		self.x_lim = x_lim
		self.max_steps = max_steps
		self.world_size = world_size

		self.continuous = continuous
		
		self.generate_all()

	def generate_all(self): 

		self.ball = Ball(lim = self.x_lim)
		self.bar = JointBar()

		self.space = pymunk.Space()
		self.space.gravity = (0,-900)
		self.space.add(self.ball.body, self.ball.shape)
		self.space.add(self.bar.body, self.bar.shape, self.bar.constraint, self.bar.joint_center)

		self.steps = 0

	def observe(self): 

		state = []
		state.append(self.ball.body.position.x/self.world_size)
		state.append(self.ball.body.position.y/self.world_size)
		state.append(self.ball.body.velocity.x/100.)
		state.append(self.ball.body.velocity.y/100.)
		state.append(self.bar.body.angle%(np.pi))
		state.append(self.bar.body.angular_velocity)
		
		return state

	def step(self,value, dt = 1/60.): 

		if self.continuous:
			self.bar.body.apply_force_at_world_point(v2(0,6000*value), (250,0))
		else: 
			action = value-1.
			self.bar.body.apply_force_at_world_point(v2(0,6000*action), (250,0))
		
		self.steps += 1 
		self.space.step(dt)

		state = self.observe()

		distance_to_center = 100*(state[0]*1. - 0.5)**2
		reward = np.exp(-distance_to_center)

		complete,reward = self.check_completion(reward)

		return state, reward, complete

	def check_completion(self, reward): 

		ball_pos = self.ball.body.position
		complete = False 
		if ball_pos.y < 250:
			complete = True 
			reward = -1.
		if self.steps > self.max_steps: 
			complete = True
			reward = 2.

		return complete,reward

	def reset(self): 

		self.generate_all()
		return self.observe()

	def infos(self): 

		if self.continuous: 
			return [len(self.observe()),1]
		else: 
			return [len(self.observe()),3]


class Render(arcade.Window): 

	def __init__(self, world, agent = None, size = 700): 

		arcade.Window.__init__(self,size,size,"Tilter")
		# input()
		self.size = size 
		self.world = world
		self.world.reset()
		self.continuous = self.world.continuous
		self.agent = agent

		self.debug_reward = 0.
		self.debug_action = 0.
		self.debug_probs = None

	def on_draw(self): 	

		arcade.start_render()

		ball_pos = self.world.ball.body.position
		ball_size = self.world.ball.size

		bar_body = self.world.bar.body.position
		bar_l = bar_body + self.world.bar.shape.a.rotated(self.world.bar.body.angle)
		bar_r = bar_body + self.world.bar.shape.b.rotated(self.world.bar.body.angle)

		arcade.draw_circle_filled(ball_pos[0], ball_pos[1], ball_size,(250,0,0))
		arcade.draw_line(bar_l[0], bar_l[1], bar_r[0], bar_r[1], (0,250,0), 5)

		arcade.draw_text("Reward {}\n Action {}".format(self.debug_reward,self.debug_action), 550, 650, arcade.color.WHITE,12)

		if self.debug_probs is not None: 
			nb = self.debug_probs.shape[0]
			x_ini, y_ini, inc = 450, 100, 20
			max_y = 150
			x = x_ini
			for i in range(nb): 
				p1 = v2(x + i*inc, y_ini)
				p2 = v2(x + (i+1)*inc, y_ini)
				p3 = p2 + v2(0, max_y*self.debug_probs[i])
				p4 = p3 + v2(-inc, 0)
				arcade.draw_polygon_filled((p1,p2,p3,p4), arcade.color.GREEN)
				arcade.draw_text("A{}".format(i), p1[0], p1[1]-20, arcade.color.WHITE,12)
				x += inc*1.1

	def update(self, value, dt = 1./60): 
		
		if self.agent != None: 
			state = self.world.observe()
			if not self.continuous: 
				self.debug_action, self.debug_probs = self.agent.demo_non_tensor(state)
		else: 
			self.debug_action = np.random.randint(3) if self.continuous else np.random.uniform(-1,1)

		_,self.debug_reward,done = self.world.step(value = value, dt = dt)
		if done: 
			self.world.reset()
		


# w = World()
# # s = w.reset()
# # for i in range(100): 
# # 	w.step(1)
# r = Render(w)
# arcade.run()

