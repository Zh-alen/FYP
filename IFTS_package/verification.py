# -*- coding: UTF-8 -*-
 
######################################
import binascii
import hashlib
import os
import sys
import time

######################################

Des_path = input('Please enter the file path: ')

######################################
def main(path):
	## 校验md5
	verify_md5_file(path)
 
def get_file_md5(file):
	m = hashlib.md5()
	with open(file, mode='rb') as f:
		while True:
			data = f.read(10240)
			if not data:
				break
			m.update(data)
	return m.hexdigest().upper()
 
def get_file_sha1(file):
	m = hashlib.sha1()
	with open(file, mode='rb') as f:
		while True:
			data = f.read(10240)
			if not data:
				break
			m.update(data)
	return m.hexdigest().upper()
 
def get_file_sha256(file):
	m = hashlib.sha256()
	with open(file, mode='rb') as f:
		while True:
			data = f.read(10240)
			if not data:
				break
			m.update(data)
	return m.hexdigest().upper()
 
def get_file_crc32(file):
	blocksize = 1024 * 64 
	with open(file, mode='rb') as f:
		crc = 0 
		while True:
			data = f.read(blocksize)
			if not data:
				break			
			crc = binascii.crc32(data, crc) 
	return hex(crc)[2:].upper()
 
def verify_md5_file(file):
	
	print(file)
	md5_file_name = file + '.md5'
	if not os.path.exists(md5_file_name) :
		print('The md5 file is not exist !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
	with open(md5_file_name, mode= 'r') as f:
		file_md5_old = f.readline().strip()
		file_sha256_old = f.readline().strip()
	file_md5 = get_file_md5(file)
	print('md5             : %s' % file_md5)
	print('md5_original    : %s' % file_md5_old)
	file_sha256 = get_file_sha256(file)
	print('sha256          : %s' % file_sha256)
	print('sha256_original : %s' % file_sha256_old)
	if file_md5 != file_md5_old or file_sha256 != file_sha256_old :
		print('The file is damage !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
	else:
		print('Verification passed !')

# ========================================================
main(Des_path)
 