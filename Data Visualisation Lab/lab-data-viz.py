# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
..
	/////////////////////////////////////////////////////////////////////////
	//
	// (c) Copyright University of Southampton, 2019
	//
	// Copyright in this software belongs to IT Innovation Centre of
	// Gamma House, Enterprise Road, Southampton SO16 7NS, UK.
	//
	// This software may not be used, sold, licensed, transferred, copied
	// or reproduced in whole or in part in any manner or form or in or
	// on any media by any person other than in accordance with the terms
	// of the Licence Agreement supplied with the software, or otherwise
	// without the prior written consent of the copyright owners.
	//
	// This software is distributed WITHOUT ANY WARRANTY, without even the
	// implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
	// PURPOSE, except where stated in the Licence Agreement supplied with
	// the software.
	//
	// Created By : Stuart E. Middleton
	// Created Date : 2019/10/31
	// Created for Project: Teaching
	//
	/////////////////////////////////////////////////////////////////////////
	//
	// Dependancies: None
	//
	/////////////////////////////////////////////////////////////////////////
	'''

Data viz lab solution

Pre-requisites
- dataset >> 20_newsgroups_corpus.json

Linux install
- py -m pip install pandas
- py -m pip install matplotlib
- py -m pip install geopandas
- py -m pip install descartes
- py -m pip install rtree

Windows install (download precompiled binaries from gohlke)
- py -m pip install pandas
- py -m pip install matplotlib
- py -m pip install Shapely-1.6.4.post2-cp37-cp37m-win_amd64.whl
- py -m pip install GDAL-3.0.1-cp37-cp37m-win_amd64.whl
- py -m pip install Fiona-1.8.9-cp37-cp37m-win_amd64.whl
- py -m pip install geopandas
- py -m pip install descartes
- py -m pip install Rtree-0.8.3-cp37-cp37m-win_amd64.whl

https://www.lfd.uci.edu/~gohlke/pythonlibs/
http://matplotlib.org/3.1.1/users/installing.html
http://pandas.pydata.org/pandas-docs/stable/install.html
http://geopandas.org/install.html


"""

import os, sys, logging, traceback, codecs, datetime, copy, time, ast, math, re, random, shutil, json, csv, multiprocessing, subprocess
import pandas as pd, geopandas as gpd
import matplotlib.pyplot as plt

def index_raw_dataset( dataset_raw = {}, group_name = None ) :

	# compute global freq of each NER type's value instance
	dict_NER_instance_freq = {}
	for dict_post in dataset_raw :
		if (group_name == None) or (dict_post['post:group'] == group_name) :
			for key in dict_post :
				if key.startswith('NER:') == True :
					if not key in dict_NER_instance_freq :
						dict_NER_instance_freq[key] = {}
					for str_NER_value in dict_post[key] :
						if not str_NER_value in dict_NER_instance_freq[key] :
							dict_NER_instance_freq[key][str_NER_value] = 0
						dict_NER_instance_freq[key][str_NER_value] = dict_NER_instance_freq[key][str_NER_value] + 1

	# compute for each NER type a global value index (sorted by freq)
	# this maps each categorical NER value to an integer index (0..N_value_instances)
	dict_NER_value_index = {}
	for str_NER_type in dict_NER_instance_freq :

		# create a sorted list of (value_instance, freq) tuples for each NER type
		list_value_freq = []
		for str_NER_value in dict_NER_instance_freq[str_NER_type] :
			list_value_freq.append( ( str_NER_value, dict_NER_instance_freq[str_NER_type][str_NER_value] ) )
		list_value_freq = sorted( list_value_freq, key=lambda entry: entry[1], reverse=True )

		# create a list of NER value indexes for each NER type (sorted by freq order to allow us later to visualize topN only)
		list_label_index = ['none']*len(list_value_freq)
		nIndex = 0
		for ( str_NER_value, nFreq ) in list_value_freq :
			list_label_index[nIndex] = str_NER_value
			nIndex = nIndex + 1
		dict_NER_value_index[str_NER_type] = list_label_index

	# get the group values in the dataset
	set_groups = set([])
	for dict_post in dataset_raw :
		str_group = dict_post['post:group']
		set_groups.add( str_group )

	return ( dict_NER_instance_freq, dict_NER_value_index, set_groups )

def prepare_dataframe_for_viz3( dataset_raw = {}, group_name = None, ner_type = None, top_n = None ) :

	# index raw dataset using group name filter
	( dict_NER_instance_freq, dict_NER_value_index, set_groups ) = index_raw_dataset( dataset_raw = dataset_raw, group_name = group_name )

	# prepare dataframe
	dict_dataframe = {}
	for str_group in set_groups :
		if (group_name == None) or (str_group == group_name) :

			for str_NER_type in dict_NER_value_index :
				if (ner_type == None) or (str_NER_type == ner_type) :

					# get global value instance index (sorted in global freq order)
					# this means we are working with integer indexes to categorical values, not string labels
					list_label_index = copy.copy( dict_NER_value_index[str_NER_type] )

					# only keep global top N value instances to avoid visualizing 1,000's of NER instance values
					if top_n != None :
						list_label_index = list_label_index[:top_n]

					# make a freq vector from NER value instance occurances in the raw dataset
					list_freq = [0]*len(list_label_index)
					for dict_post in dataset_raw :
						if (dict_post['post:group'] == str_group) and (str_NER_type in dict_post) :
							list_NER_values = dict_post[str_NER_type]
							for str_NER_value in list_NER_values :
								if str_NER_value in list_label_index :
									nIndex = list_label_index.index( str_NER_value )
									list_freq[nIndex] = list_freq[nIndex] + 1

					# add to data structure for viz
					dict_dataframe[ str_NER_type ] = list_freq
					dict_dataframe[ str_NER_type + '_label' ] = list_label_index

	return dict_dataframe

def prepare_dataframe_for_viz2( dataset_raw = {}, group_name = None ) :

	# index raw dataset using group name filter
	( dict_NER_instance_freq, dict_NER_value_index, set_groups ) = index_raw_dataset( dataset_raw = dataset_raw, group_name = group_name )

	# prepare dataframe
	dict_dataframe = {}

	# get global NER type index
	# this means we are working with integer indexes to categorical values, not string labels
	list_label_index = list( dict_NER_value_index.keys() )

	# make a freq vector
	list_freq = [0]*len(list_label_index)

	for dict_post in dataset_raw :
		if (dict_post['post:group'] == group_name) or (group_name == None) :
			for key in dict_post :
				if key.startswith('NER:') == True :
					nIndex = list_label_index.index( key )
					list_freq[nIndex] = list_freq[nIndex] + len( dict_post[key] )

	# add to data structure for viz
	dict_dataframe[ 'NER_type' ] = list_freq
	dict_dataframe[ 'NER_type_label' ] = list_label_index

	return dict_dataframe

def prepare_dataframe_for_viz1( dataset_raw = {} ) :

	# index raw dataset using group name filter
	( dict_NER_instance_freq, dict_NER_value_index, set_groups ) = index_raw_dataset( dataset_raw = dataset_raw, group_name = None )

	# prepare dataframe
	dict_dataframe = {}

	# make a global instance index for each group
	# this means we are working with integer indexes to categorical values, not string labels
	list_label_index = list( set_groups )

	# make a post freq vector for each group
	list_freq = [0]*len(list_label_index)
	for str_group in set_groups :
		for dict_post in dataset_raw :
			if dict_post['post:group'] == str_group :
				nIndex = list_label_index.index( str_group )
				list_freq[nIndex] = list_freq[nIndex] + 1

	# add to data structure for viz
	dict_dataframe[ 'groups' ] = list_freq
	dict_dataframe[ 'groups_label' ] = list_label_index

	return dict_dataframe

def prepare_dataframe_for_viz4( dataset_raw = {} ) :

	# index raw dataset using group name filter
	( dict_NER_instance_freq, dict_NER_value_index, set_groups ) = index_raw_dataset( dataset_raw = dataset_raw, group_name = None )

	# prepare dataframe
	dict_dataframe = {}

	# make a set of post length range bins
	list_range_max = [ 5,10,20,50,100,200,500,1000,2000,5000 ]
	list_label_index = []
	str_previous = '< '
	for entry in list_range_max :
		list_label_index.append( str_previous + str(entry) )
		str_previous = str(entry) + ' to '
	list_label_index[-1] = '5000+'

	# make a post freq vector for each group
	list_freq = [0]*len(list_label_index)
	for dict_post in dataset_raw :
		nTextLength = len( dict_post['body:text'] )

		# work out the range bin text length should fall into
		nIndexBin = len(list_range_max) - 1
		for nIndex in range(len(list_range_max)) :
			if nTextLength < list_range_max[nIndex] :
				nIndexBin = nIndex
				break

		# add to freq for that bin
		list_freq[nIndexBin] = list_freq[nIndexBin] + 1

	# add to data structure for viz
	dict_dataframe[ 'post_length' ] = list_freq
	dict_dataframe[ 'post_length_label' ] = list_label_index

	return dict_dataframe

def load_raw_dataset( filename = None ) :

	read_handle = codecs.open( filename, 'r', 'utf-8', errors = 'replace' )
	list_lines = read_handle.readlines()
	read_handle.close()

	dataset_raw = []
	for str_line in list_lines :
		dict_post = json.loads( str_line )
		dataset_raw.append( dict_post )

	return dataset_raw

################################
# main
################################

# only execute if this is the main file
if __name__ == '__main__' :

	# make logger (global to STDOUT)
	LOG_FORMAT = ('%(levelname) -s %(asctime)s %(message)s')
	logger = logging.getLogger( __name__ )
	logging.basicConfig( level=logging.INFO, format=LOG_FORMAT )
	logger.info('logging started')

	#
	# generic histogram plot using pandas
	#

	dict_dataset = { 'category_freq' : [0,1,5,6], 'category_labels' : [' A', 'B', 'C', 'D'] }
	df = pd.DataFrame( data = dict_dataset )
	df.plot.barh( x='category_labels', y='category_freq', alpha=0.5, title = 'Category freq breakdown' )
	plt.show()

	#
	# load raw dataset
	#

	strFile = '20_newsgroups_corpus.json'
	logger.info( 'loading dataset: ' + strFile )
	listDatasetRaw = load_raw_dataset( filename = strFile )
	logger.info( 'Number of posts in raw dataset = ' + repr(len(listDatasetRaw)) )

	#
	# visualize post length breakdown (all groups)
	#

	dictDataToViz = prepare_dataframe_for_viz4(
		dataset_raw = listDatasetRaw )
	
	df = pd.DataFrame( data = dictDataToViz )
	df.plot.barh( x='post_length_label', y='post_length', alpha=0.5, title = 'Post length breakdown (all groups)' )
	plt.show()

	#
	# visualize post freq breakdown for groups
	# tutorial (all): https://pandas.pydata.org/pandas-docs/version/0.23/tutorials.html
	# tutorial (bar chart): https://pandas.pydata.org/pandas-docs/version/0.23/generated/pandas.DataFrame.plot.barh.html
	#

	dictDataToViz = prepare_dataframe_for_viz1(
		dataset_raw = listDatasetRaw )
	
	df = pd.DataFrame( data = dictDataToViz )
	df.plot.barh( x='groups_label', y='groups', alpha=0.5, title = 'Post freq by group' )
	plt.show()

	#
	# visualize NER type mention freq breakdown for all groups
	#

	dictDataToViz = prepare_dataframe_for_viz2(
		dataset_raw = listDatasetRaw,
		group_name = None )

	df = pd.DataFrame( data = dictDataToViz )
	df.plot.barh( x='NER_type_label', y='NER_type', alpha=0.5, title = 'NER freq for all groups' )
	plt.show()

	#
	# visualize NER type value mention freq breakdown for talk.politics.misc
	#

	dictDataToViz = prepare_dataframe_for_viz2(
		dataset_raw = listDatasetRaw,
		group_name = 'talk.politics.misc' )

	df = pd.DataFrame( data = dictDataToViz )
	df.plot.barh( x='NER_type_label', y='NER_type', alpha=0.5, title = 'NER freq for talk.politics.misc' )
	plt.show()

	#
	# visualize all NER:COUNTRY value breakdown for talk.politics.misc
	#
	dictDataToViz = prepare_dataframe_for_viz3(
		dataset_raw = listDatasetRaw,
		group_name = 'talk.politics.misc',
		ner_type = 'NER:COUNTRY',
		top_n = 50 )

	df = pd.DataFrame( data = dictDataToViz )
	df.plot.barh( x='NER:COUNTRY_label', y='NER:COUNTRY', alpha=0.5, title = 'NER:COUNTRY value freq for talk.politics.misc' )
	plt.show()

	#
	# generic geopandas plot
	#

	world = gpd.read_file( gpd.datasets.get_path('naturalearth_lowres') )
	world = world[(world.pop_est>0) & (world.name!="Antarctica")]
	world['gdp_per_cap'] = world.gdp_md_est / world.pop_est
	world.plot( column='gdp_per_cap' );
	plt.show()

	for index, row in world.iterrows():
		str_place = world.loc[ index, 'name' ]
		if str_place == 'United Kingdom' :
			world.loc[ index,'favourite' ] = 1
		else :
			world.loc[ index,'favourite' ] = 0

	world.plot( column='favourite' );
	plt.show()

	#
	# visualize all NER:COUNTRY values on a map
	# tutorial (all): https://geopandas.readthedocs.io/en/latest/index.html
	# tutorial (mapping): https://geopandas.readthedocs.io/en/latest/mapping.html
	#

	dictDataToViz = prepare_dataframe_for_viz3(
		dataset_raw = listDatasetRaw,
		group_name = 'talk.politics.misc',
		ner_type = 'NER:COUNTRY',
		top_n = None )

	world = gpd.read_file( gpd.datasets.get_path('naturalearth_lowres') )
	world = world[ (world.geometry != None) ]

	for index, row in world.iterrows():

		strPlace = world.loc[ index, 'name' ]

		nFreqLoc = 0
		for strLocationValue in dictDataToViz[ 'NER:COUNTRY_label' ] :
			if strLocationValue.lower() == strPlace.lower() :
				nIndexLoc = dictDataToViz[ 'NER:COUNTRY_label' ].index( strLocationValue )
				nFreqLoc = dictDataToViz[ 'NER:COUNTRY' ][ nIndexLoc ]
				break

		world.loc[ index,'post_freq' ] = nFreqLoc

	world.plot( column='post_freq', legend=True );
	plt.show()

