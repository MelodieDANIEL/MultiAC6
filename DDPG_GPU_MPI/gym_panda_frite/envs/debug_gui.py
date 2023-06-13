import pybullet as p

class Debug_Gui:
	
	def __init__(self, env = None):
		self.env = env
		self.dic_id = {}
	
	def reset(self):
		self.dic_id = {}
	
	def draw_text(self, a_name, a_text = "", a_pos = [0,0,0], a_size = 1.5, a_color = [1, 0, 0]):
		if ( str(a_name)+"_txt" in self.dic_id.keys() ):
			p.addUserDebugText(a_text, a_pos,textColorRGB=a_color,textSize=a_size,replaceItemUniqueId  = self.dic_id[str(a_name)+"_txt"])
		else:
			self.dic_id[str(a_name)+"_txt"] = p.addUserDebugText(a_text, a_pos,textColorRGB=a_color,textSize=a_size)
		
	def remove_text(self, a_name):
		p.removeUserDebugItem(self.dic_id[str(a_name)+"_txt"])
		del self.dic_id[str(a_name)+"_txt"]
		
	def draw_line(self, name, a_pos_from = [0,0,0], a_pos_to = [0,0,0], a_size = 0.1, a_color = [1, 0, 0], a_width = 3.0, a_time = 0):
		if ( str(name)+"_line" in self.dic_id.keys() ):
			p.addUserDebugLine(lineFromXYZ      = a_pos_from  ,
			   lineToXYZ            = a_pos_to,
			   lineColorRGB         = a_color  ,
			   replaceItemUniqueId  = self.dic_id[str(name)+"_line"],
			   lineWidth            = a_width        ,
			   lifeTime             = a_time
					 )	
		else:
			self.dic_id[str(name)+"_line"] = p.addUserDebugLine(lineFromXYZ      = a_pos_from  ,
			   lineToXYZ            = a_pos_to,
			   lineColorRGB         = a_color  ,
			   lineWidth            = a_width        ,
			   lifeTime             = a_time
					 )	
		
	def remove_line(self, a_name):
		p.removeUserDebugItem(self.dic_id[str(a_name)+"_line"])
		del self.dic_id[str(a_name)+"_line"]
		
	def draw_cross(self, name, a_pos = [0,0,0], a_size = 0.1, a_color = [1, 0, 0], a_width = 3.0, a_time = 0):
		
		if ( str(name)+"_x" in self.dic_id.keys() ):
			p.addUserDebugLine(lineFromXYZ      = [a_pos[0]+a_size, a_pos[1], a_pos[2]]  ,
			   lineToXYZ            = [a_pos[0]-a_size, a_pos[1], a_pos[2]],
			   lineColorRGB         = a_color,
			   replaceItemUniqueId  = self.dic_id[str(name)+"_x"],
			   lineWidth            = a_width,
			   lifeTime             = a_time
				  )
		else:
			self.dic_id[str(name)+"_x"] = p.addUserDebugLine(lineFromXYZ      = [a_pos[0]+a_size, a_pos[1], a_pos[2]]  ,
			   lineToXYZ            = [a_pos[0]-a_size, a_pos[1], a_pos[2]],
			   lineColorRGB         = a_color,
			   lineWidth            = a_width,
			   lifeTime             = a_time
				  )

		if ( str(name)+"_y" in self.dic_id.keys() ):
			p.addUserDebugLine(lineFromXYZ      = [a_pos[0], a_pos[1]+a_size, a_pos[2]]  ,
			   lineToXYZ            = [a_pos[0], a_pos[1]-a_size, a_pos[2]],
			   lineColorRGB         = a_color  ,
			   replaceItemUniqueId  = self.dic_id[str(name)+"_y"],
			   lineWidth            = a_width        ,
			   lifeTime             = a_time
					 )
			
		else:
			self.dic_id[str(name)+"_y"] = p.addUserDebugLine(lineFromXYZ      = [a_pos[0], a_pos[1]+a_size, a_pos[2]]  ,
			   lineToXYZ            = [a_pos[0], a_pos[1]-a_size, a_pos[2]],
			   lineColorRGB         = a_color  ,
			   lineWidth            = a_width        ,
			   lifeTime             = a_time
					 )
			
		
		if ( str(name)+"_z" in self.dic_id.keys() ):
			p.addUserDebugLine(lineFromXYZ      = [a_pos[0], a_pos[1], a_pos[2]+a_size]  ,
			   lineToXYZ            = [a_pos[0], a_pos[1], a_pos[2]-a_size],
			   lineColorRGB         = a_color  ,
			   replaceItemUniqueId  = self.dic_id[str(name)+"_z"],
			   lineWidth            = a_width        ,
			   lifeTime             = a_time
					 )
		else:
			self.dic_id[str(name)+"_z"] = p.addUserDebugLine(lineFromXYZ      = [a_pos[0], a_pos[1], a_pos[2]+a_size]  ,
			   lineToXYZ            = [a_pos[0], a_pos[1], a_pos[2]-a_size],
			   lineColorRGB         = a_color  ,
			   lineWidth            = a_width        ,
			   lifeTime             = a_time
					 )
		
	def remove_cross(self, a_name):
		p.removeUserDebugItem(self.dic_id[str(a_name)+"_x"])
		del self.dic_id[str(a_name)+"_x"]
		p.removeUserDebugItem(self.dic_id[str(a_name)+"_y"])
		del self.dic_id[str(a_name)+"_y"]
		p.removeUserDebugItem(self.dic_id[str(a_name)+"_z"])
		del self.dic_id[str(a_name)+"_z"]

	def draw_box(self, name, low , high , color = [0, 0, 1]):
		low_array = low
		high_array = high

		p1 = [low_array[0], low_array[1], low_array[2]] # xmin, ymin, zmin
		p2 = [high_array[0], low_array[1], low_array[2]] # xmax, ymin, zmin
		p3 = [high_array[0], high_array[1], low_array[2]] # xmax, ymax, zmin
		p4 = [low_array[0], high_array[1], low_array[2]] # xmin, ymax, zmin
		
		self.draw_line(str(name)+"_p1_p2" , a_pos_from = p1, a_pos_to = p2, a_size = 0.1, a_color = color, a_width = 2.0, a_time = 0)
		self.draw_line(str(name)+"_p2_p3" , a_pos_from = p2, a_pos_to = p3, a_size = 0.1, a_color = color, a_width = 2.0, a_time = 0)
		self.draw_line(str(name)+"_p3_p4" , a_pos_from = p3, a_pos_to = p4, a_size = 0.1, a_color = color, a_width = 2.0, a_time = 0)
		self.draw_line(str(name)+"_p4_p1" , a_pos_from = p4, a_pos_to = p1, a_size = 0.1, a_color = color, a_width = 2.0, a_time = 0)

		p5 = [low_array[0], low_array[1], high_array[2]] # xmin, ymin, zmax
		p6 = [high_array[0], low_array[1], high_array[2]] # xmax, ymin, zmax
		p7 = [high_array[0], high_array[1], high_array[2]] # xmax, ymax, zmax
		p8 = [low_array[0], high_array[1], high_array[2]] # xmin, ymax, zmax

		self.draw_line(str(name)+"_p5_p6" , a_pos_from = p5, a_pos_to = p6, a_size = 0.1, a_color = color, a_width = 2.0, a_time = 0)
		self.draw_line(str(name)+"_p6_p7" , a_pos_from = p6, a_pos_to = p7, a_size = 0.1, a_color = color, a_width = 2.0, a_time = 0)
		self.draw_line(str(name)+"_p7_p8" , a_pos_from = p7, a_pos_to = p8, a_size = 0.1, a_color = color, a_width = 2.0, a_time = 0)
		self.draw_line(str(name)+"_p8_p5" , a_pos_from = p8, a_pos_to = p5, a_size = 0.1, a_color = color, a_width = 2.0, a_time = 0)
		
		self.draw_line(str(name)+"_p1_p5" , a_pos_from = p1, a_pos_to = p5, a_size = 0.1, a_color = color, a_width = 2.0, a_time = 0)
		self.draw_line(str(name)+"_p2_p6" , a_pos_from = p2, a_pos_to = p6, a_size = 0.1, a_color = color, a_width = 2.0, a_time = 0)
		self.draw_line(str(name)+"_p3_p7" , a_pos_from = p3, a_pos_to = p7, a_size = 0.1, a_color = color, a_width = 2.0, a_time = 0)
		self.draw_line(str(name)+"_p4_p8" , a_pos_from = p4, a_pos_to = p8, a_size = 0.1, a_color = color, a_width = 2.0, a_time = 0)

	def remove_box(self, a_name):
		self.remove_line(str(a_name)+"_p1_p2")
		self.remove_line(str(a_name)+"_p2_p3")
		self.remove_line(str(a_name)+"_p3_p4")
		self.remove_line(str(a_name)+"_p4_p1")
		
		self.remove_line(str(a_name)+"_p5_p6")
		self.remove_line(str(a_name)+"_p6_p7")
		self.remove_line(str(a_name)+"_p7_p8")
		self.remove_line(str(a_name)+"_p8_p5")
		
		self.remove_line(str(a_name)+"_p1_p5")
		self.remove_line(str(a_name)+"_p2_p6")
		self.remove_line(str(a_name)+"_p3_p7")
		self.remove_line(str(a_name)+"_p4_p8")
