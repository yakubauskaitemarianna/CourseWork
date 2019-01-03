# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 12:10:55 2018

@author: Rin
"""
import json
import sys
import xml.dom.minidom
from collections import defaultdict

def xmltodict(xmlstring):
    doc = xml.dom.minidom.parseString(xmlstring)
    return elementtodict(doc)

def elementtodict(parent):
    child = parent.firstChild
    while child and child.nodeType == xml.dom.minidom.Node.TEXT_NODE and not child.data.strip():
        child = child.nextSibling
    if not child:
        return None
    if child.nodeType == xml.dom.minidom.Node.TEXT_NODE or child.nodeType == xml.dom.minidom.Node.CDATA_SECTION_NODE:
        value = child.nodeValue
        if value.isdigit():
            value = int(value)
        return value
    d = defaultdict(list)
    while child:
        if child.nodeType == xml.dom.minidom.Node.ELEMENT_NODE:
            attr_dict = {}
            if child.hasAttributes():
                attrs = child.attributes
                for i in range(0, attrs.length):
                    _attr = attrs.item(i)
                    attr_dict[_attr.name] = _attr.value
            d[child.tagName].append({'attr' : attr_dict, 'child' : elementtodict(child)})
        child = child.nextSibling
    return dict(d)
