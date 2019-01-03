# -*- coding: utf-8 -*-
"""
Created on Sat Jul 14 19:10:55 2018

@author: Rin
"""
import xmlparse
from xmlparse import xmltodict
from PIL import Image, ImageDraw
import os


s = open(xml_file, encoding="utf8").read()
d = xmlparse.xmltodict(s)

attributelst = [] 
sizes = []
page_data = []

def parse_dict(d):
    for key, value in d.items():
        for i in range(len(value)):
            for k, v in value[i].items():
                attributelst.append(v)
                if len(attributelst) > 1 and type(attributelst[1]) == dict:
                    chars = attributelst[1].get('characters')
                    pages = attributelst[1].get('pages')
                    init_page = ((pages[0]).get('child')).get('page')
                    for p in range(len(init_page)):
                        sizes.append(init_page[p].get('attr'))
                        page_data.append(init_page[p].get("child"))
                        

def get_arrays_for_draw(sizes, page_data, frames, bodies, faces):
    for i in range(len(page_data)):
        if type(page_data[i]) == dict:
            
            if type(page_data[i].get('frame')) != type(None):
                for frame in page_data[i].get('frame'):
                    xmin = (frame.get('attr')).get('xmin')
                    ymin = (frame.get('attr')).get('ymin')
                    xmax = (frame.get('attr')).get('xmax')
                    ymax = (frame.get('attr')).get('ymax')
                    frames.append([xmin, ymin, xmax, ymax])
                    
            if type(page_data[i].get('body')) != type(None):
                for body in page_data[i].get('body'):
                    xmin = (body.get('attr')).get('xmin')
                    ymin = (body.get('attr')).get('ymin')
                    xmax = (body.get('attr')).get('xmax')
                    ymax = (body.get('attr')).get('ymax')
                    bodies.append([xmin, ymin, xmax, ymin])
                
            if type(page_data[i].get('face')) != type(None):
                for face in page_data[i].get('face'):
                    xmin = (face.get('attr')).get('xmin')
                    ymin = (face.get('attr')).get('ymin')
                    xmax = (face.get('attr')).get('xmax')
                    ymax = (face.get('attr')).get('ymax')
                    faces.append([xmin, ymin, xmax, ymin])
               

parse_dict(d)
frames = []
bodies = []
faces = []
get_arrays_for_draw(sizes, page_data, frames, bodies, faces)  
