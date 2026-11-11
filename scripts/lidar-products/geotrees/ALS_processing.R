# Processing pipeline for Airborne Laser Scanning (ALS) 
#
# Produces standardized sets of digital terrain models (DTMs), digital surface
# models (DSMs), canopy height models (CHMs). The latter include rasterizations 
# of highest returns (chm_highest.tif), Delaunay-triangulation (chm_tin.tif), 
# and a locally adaptive spikefree algorithm that provides the most robust
# representation of top canopy height (chm_lspikefree.tif). The script also
# generates ancillary rasters for data quality control (pulse density, 
# scan angle, laser penetration), and outline shapefiles with data set 
# information. The script will run through a set of directories, process 
# multiple data sets and write them back to the output directory in the same 
# structure as in the input directory.
#
#
# DEVELOPERS: 
#
#   Fabian Jörg Fischer and Tommaso Jucker",
#
# ADDITIONAL CONTRIBUTORS TO DEVELOPMENT:",
#
#   Toby Jackson, Greg Vincent, Becky Morgan, Nicolas Labriere",
#   Andres Gonzalez-Moreno, Jerome Chave, Maxime Rejou-Mechain, Karl Montalban",
#
# CITATION:
#
#   Fischer, F. J., Jackson, T., Vincent, G., & Jucker, T. 2024.",
#   Robust characterisation of forest structure from airborne laser",
#   scanning — A systematic assessment and sample workflow for ecologists.",
#   Methods in Ecology and Evolution. https://doi.org/10.1111/2041-210"
#
# GENERAL TIPS FOR PROCESSING:
#
# 1. A metadata file (_metadata.csv) should be provided: this file will be used 
# to update information that can not be directly extracted from the scan. Key 
# information is: data citation/website/contact + coordinate reference system
# (EPSG code) + temporal acquisition range (minimum and maximum date). An 
# example is provided with the test data.
#
# 2. The folder structure in which las/laz files are stored should reflect 
# the structure of the underlying acquisitions, i.e., a folder should only 
# contain las/laz files that have been acquired at the same site and during the 
# same time period and in the same coordinate system. The pipeline will separate
# spatially distinct geo-units, but not check temporal ranges.
# 
# 3. None of the data files (laz/las) should have path lengths > 260 characters, 
# particularly when using multiple nested folders. Oddly, copy operations will 
# fail on Windows when this limit is exceeded, and there does not seem to be an 
# easy work-around.
#
# 4. The temporary folder (tmpdir) where data are processed should never be on a 
# synchronized drive or external hard drive, otherwise processing might be slow
#
# 5. When multi-processing, the maximum number of cores should be determined by 
# the number of physical cores, not logical ones. The script will check this and
# throw a warning if not true. However, this warning can be safely ignored in 
# some cases where (virtual machines, cloud processing)
#
#
#
# FOR MORE INFORMATION check _INFO_ files produced at the end of processing and
# the reference publication cited above.

#%%%%%%%%%%%%%%%%%%#
##### 1. Setup #####
#%%%%%%%%%%%%%%%%%%#
# use up-to-date versions of R version, R packages and LAStools! 
# update.packages()

# set working directory  
setwd(here::here())

# load helper functions
file_helperfunctions = "~/R/Rstudios/Saopaulo/02_ancillary/ALS_processing_helperfunctions.R"

source(file_helperfunctions)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
##### 2. Processing parameters (fixed tile size) #####
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
# standard processing script, using a fixed tile size
name_job = "gca"
type_file = "las"
dir_dataset = "C:/Users/rodri/Documents/R/Rstudios/Saopaulo/01_raw"
dir_processed =  "C:/Users/rodri/Documents/R/Rstudios/Saopaulo/03_processed/"
tmpdir_processing = "C:/Users/rodri/Documents/R/Rstudios/Saopaulo/TMPDIR/" 
path_lastools = ""
resolution = 1.0
n_cores = 10
size_tile = 100
size_buffer = 50
force.utm = "from_metadata"          # force reprojection of system into UTM (and meter) coordinates; necessary for all files that are registered in feet, otherwise output will be in feet
force.recompute = F  # Set to TRUE to ignore any checkpoint and reprocess everything from scratch    # force reprocessing; usually set to FALSE, useful when computation has been interrupted for external reasons (power cutofff) and needs to be restarted, because only unprocessed data subsets will be reprocessed
remove.vlr = F    # probably not necessary in most cases, but should be generally activated. The option leads to the removal of all vlrs AFTER the CRS of the first raster product is set. The CRS will be transmitted to all other raster products, so outputs will not be affected. By deactivating vlrs afterwards we prevent problems with further terra or lastools processing due to odd projection information, which sometimes causes an excess of warnings/errors (and shutdown of parallel processing) or blast2dem to fail
remove.evlr = F   # probably not necessary in most cases, but should be generally activated. The option leads to the removal of all evlrs AFTER the CRS of the first raster product is set. The CRS will be transmitted to all other raster products, so outputs will not be affected. By deactivating evlrs afterwards we prevent problems with further terra or lastools processing due to odd projection information, which sometimes causes an excess of warnings/errors (and shutdown of parallel processing) or blast2dem to fail
use.blast2dem = T # should be activated by default as it improves (or makes possible) TIN construction in very dense point clouds, but: not tested on Linux so far
type_architecture = "64" # only needed if 32bit Windows should be forced; 32 can be a bit more permissive, particularly for blast2dem


results = process.dataset(
  name_job = name_job
  , type_file = type_file
  , dir_dataset = dir_dataset
  , dir_processed = dir_processed
  , tmpdir_processing = tmpdir_processing
  , path_lastools = path_lastools
  , resolution = resolution
  , n_cores = n_cores
  , size_tile = size_tile
  , size_buffer = size_buffer
  , force.utm = force.utm
  , remove.vlr = remove.vlr
  , remove.evlr = remove.evlr
  , type_architecture = type_architecture
  , use.blast2dem = use.blast2dem
  , patterns_skip = c() # skipping the processing of some folders
  , cleanup = T
  , retile = F # not in the open source version
)

# # other parameters
# metadata = NULL
# cleanup = T
# nbclusters_forced = NULL
# remove.buffer = F
# factor_rescale = NULL
# path_output_lazclean = ""
# path_output_laznorm = ""
# types_dsm = c("tin","lspikefree")
# params_dsmadaptive = data.table(multi = 3.1, slope = 1.75, offset = 2.1) # do not modify, this is calibrated for maximum robustness
# resolution_sumstatspc = NULL
# add.timestamp = F
# estimate.laserpenetration = F
# type_os = "automatic"
# by_file = F
# perturbation_max = 0.1 # do not modify
# timeout_lspikefree_max = 600 # do not modify
# overwrite.crs = F
# is.stdtime = NA
# height_lim = 125
# angle_lim = NULL
# class_rm = c()
# exclass_rm = c()
# force.type_point = NULL
# logfile = ""
# patterns_skip = c()
# print.summary_job = F
# 
