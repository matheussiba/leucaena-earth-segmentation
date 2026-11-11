# this script contains helper functions to process lidar data
# the initial structure was inspired by scripts created by Toby Jackson, Tom Swinfield and Laura Bentley for LAStools processing from R
#
# this version is made as a non-lastools version of the original pipeline.
# this version is adapted from the 1.0.3 version of the GCA pipeline (with lastools functions)
# the lastools part are kept but should not be executed if the user set path_lastools = ""

library(Hmisc)
library(data.table)
library(terra)
library(sf) # for buffering polygons, sf st_buffer is more permissive/less error prone than terra's buffer function
library(igraph) # for clustering
library(viridisLite) # for visualization

library(lidR)
library(lasR) # for automatic referencing

# parallel computing
library(parallel) # for custom parallel processing
# library(foreach)
# library(doParallel)
library(future)

#%%%%%%%%%%%%%%%%%%%%%%%%%#
#### 0. Info functions ####
#%%%%%%%%%%%%%%%%%%%%%%%%%#
# these are functions that produce metadata, need to be kept up to date with code development
# we use underscores to position these files at top of directories under alphabetical ordering

make.info = function(path_output = "", name_file = "_INFO_.txt", params_general){
  
  if(path_output != ""){
    # get current version of the script
    version_pipeline = "v.1.1.0" #gsub("v","", gsub("v.","",gsub(".R","", path_script_split[length(path_script_split)][[1]], fixed = T), fixed = T))
    
    # make output file
    file_output = file.path(path_output, name_file)
    
    # Content of the INFO file
    content_info = c(
      paste0("## Processing pipeline for Airborne Laser Scanning (ALS) v.",version_pipeline),
      "",
      "   Developed as part of the Global Canopy Atlas (GCA) project.",
      "   Derives elevation and canopy height rasters that are maximally robust",
      "   to differences in instrumentation and pulse density between scans.",
      "",
      "   Executed from R. Dependent on LAStools (https://rapidlasso.de/).",
      "   License: GPLv3 (http://www.gnu.org/licenses/gpl.html)",
      paste0("   Operating System: ", params_general$type_os, "(", params_general$type_architecture,"bit)"),
      paste0("   LAStools version: ", get.version_lastools(params_general)),
      "",
      "## DEVELOPERS:",
      "",
      "   Fabian Jörg Fischer and Tommaso Jucker",
      "",
      "## ADDITIONAL CONTRIBUTORS TO DEVELOPMENT:",
      "",
      "   Toby Jackson, Greg Vincent, Becky Morgan, Nicolas Labriere",
      "   Andres Gonzalez-Moreno, Jerome Chave, Maxime Rejou-Mechain",
      "   Karl Montalban",
      "",
      "## CITATION:",
      "",
      "   Fischer, F. J., Jackson, T., Vincent, G., & Jucker, T. 2024.",
      "   Robust characterisation of forest structure from airborne laser",
      "   scanning — A systematic assessment and sample workflow for ecologists.",
      "   Methods in Ecology and Evolution. https://doi.org/10.1111/2041-210X.14416",
      "",
      "   Github: https://github.com/fischer-fjd/GCA",
      "   Zenodo: 10.5281/zenodo.13883963",
      "",
      "## PRODUCTS:",
      "",
      "   Digital surface models (DSMs)",
      "   Digital terrain models (DTMs)",
      "   Canopy height models (CHMs)", 
      "   Pulse density rasters", 
      "   Scan angle rasters",
      "   Data quality masks", 
      "   Outline shapefiles with processing summaries",
      "",
      "## CHANGES:",
      "",
      "   v.1.0.3: Bug fix ground refinement (no error when 0 ground points)",
      "   v.1.0.2: New ERROR.txt files",
      "   v.1.0.1: Option to remove RGB information via force.type_point = 1",
      "            Changed license to GPLv3 (recommended for code over CC BY 4.0)",
      "            Fixed BUG in refine.ground: focal error for tile size < 25 m",
      "   v.1.0.0: First release version, surpasses development versions (v1-50)",
      "            Linux + Windows support",
      "            Refinement of ground classification/DTMs in steep areas",
      "            Additional masks to subset to high-quality areas",
      "            Thinning at 10 cm resolution for dsm_lspikefree computation",
      "            Improved documentation, input checks and file naming",
      "",
      "## FURTHER INFO:",
      "",
      "   Individual products: _INFO_products.csv",
      "   Processing summaries: summary_processing.csv + _INFO_summary_processing.csv"
    )
    
    # Write the content to an INFO.txt file
    writeLines(content_info, con = file_output)
    # cat(name_file,"created at:", path_output,"\n")
  }
}

make.info_products = function(path_output = "",name_file = "_INFO_products.csv"){
  
  if(path_output != ""){
    # make output file
    file_output = file.path(path_output, name_file)
    # Create a data table with file names and descriptions
    products = data.table(
      file = c(
        "_INFO_.txt",
        "_INFO_products.csv",
        "_INFO_summary_processing.csv",
        "chm_highest.tif",
        "chm_lspikefree.tif",
        "chm_tin.tif",
        "classification_supplied.tif",
        "dsm_highest.tif",
        "dsm_lspikefree.tif",
        "dsm_tin.tif",
        "dtm_highest.tif",
        "dtm_lasdef.tif",
        "dtm_lasfine.tif",
        "dtm_supplied.tif",
        "dtm.tif",
        "mask_cloud.tif",
        "mask_combined.tif",
        "mask_noground.tif",
        "mask_unstabledtm.tif",
        "mask_pd02.tif",
        "mask_pd04.tif",
        "mask_steep.tif",
        "outline_localCRS.shp (cpg/dbh/prj)",
        "outline_WGS84.shp (cpg/dbh/prj)",
        "pulsedensity_lastreturn.tif",
        "pulsedensity_scanangle20.tif",
        "pulsedensity.tif",
        "scanangle_abs.tif",
        "summary_processing.csv"
      ),
      description = c(
        "Information on processing pipeline",
        "Information on generated files",
        "Information on processing summaries stored in summary_processing.csv, as well as in outline_localCRS.shp and outline_WGS84.shp",
        "Canopy height model (CHM). Canopy height estimates based on highest return per pixel (typical resolution: 1 m2), relative to ground elevation. Computed as dsm_highest.tif - dtm.tif. Useful for homogeneous, high-quality scans (>= 30 pulses per m2) and individual tree crown delineations. Not recommended for comparative analyses across time or space with varying scanning properties. Cf. recommendations in: https://doi.org/10.1111/2041-210X.14416, in particular Fig. 4.",
        "Canopy height model (CHM). Canopy height estimates based on locally adaptive spikefree algorithm applied to first returns, relative to ground elevation. Computed as dsm_lspikefree.tif - dtm.tif. Maximally robust CHM that creates a spikefree interpolation of top canopy height while adjusting to local variation in pulse density. Recommended for comparative analyses across time and space. Applicable down to densities of 2-3 pulses per m2. Limited use for individual tree delineation due to smoothing of edges. Cf. recommendations in: https://doi.org/10.1111/2041-210X.14416, in particular Fig. 4.", 
        "Canopy height model (CHM). Canopy height estimates based on Delaunay-triangulation of first returns, relative to ground elevation. Computed as dsm_tin.tif - dtm.tif. Robust to point cloud density variation, but usually full of pits and spikes. Interpretable as the mean height of light interception within the canopy. Useful in conjunction with locally adaptive spikefree CHM. Cf. recommendations in: https://doi.org/10.1111/2041-210X.14416, in particular Fig. 4.",
        "Rasterized point cloud classifications as supplied by data provider. Useful to check for special noise classifications that may have been overlooked, or for further downstream analysis (exclusion of buildings). The default LAStools rasterization choice (class of highest point) is used.",
        "Digital surface model (DSM). Surface height estimates based on highest return per pixel (typical resolution: 1 m2). Useful for homogeneous, high-quality scans (>= 30 pulses per m2) and individual tree crown delineations. Not recommended for comparative analyses across time or space with varying scanning properties. Cf. recommendations in: https://doi.org/10.1111/2041-210X.14416, in particular Fig. 4.",
        "Digital surface model (DSM). Surface height estimates based on locally adaptive spikefree algorithm applied to first returns. Maximally robust DSM that creates a spikefree interpolation of top surface height while adjusting to local variation in pulse density. Recommended for comparative analyses across time and space. Applicable down to densities of 2-3 pulses per m2. Limited use for individual tree delineation due to smoothing of edges. Cf. recommendations in: https://doi.org/10.1111/2041-210X.14416, in particular Fig. 4.", 
        "Digital surface model (DSM). Surface height estimates based on Delaunay-triangulation of first returns. Robust to point cloud density variation, but usually full of pits and spikes. Interpretable as the mean height of light interception within the canopy. Useful in conjunction with locally adaptive spikefree DSM. Cf. recommendations in: https://doi.org/10.1111/2041-210X.14416, in particular Fig. 4.",
        "Digital terrain model (DTM). Elevation estimates based on highest ground return per pixel (typical resolution: 1 m2). Not interpolated and therefore not useful for terrain or canopy modelling. Useful for quality checks, e.g. to identify large areas without ground points.",
        "Digital terrain model (DTM). Default LAStools digital terrain model. Computed from default ground classification with lasground_new and Delaunay-triangulation with las2dem. Useful as baseline and for comparisons across datasets.",
        "Digital terrain model (DTM). Refined DTM, created in LAStools with more sensitive parameters (-step 10 -bulge 1.0 -hyper_fine). Useful in conjunction with dtm_lasdef.tif to assess robustness of ground modelling.",
        "Digital terrain model (DTM). DTM supplied by data provider. May not exist. Useful for quality checks, as supplied DTMs often (but not always!) contain manual corrections.",
        "Digital terrain model (DTM). The default terrain model used by the processing pipeline. Based on the default LAStools lasground_new classification, but refined to improve accuracy in steep areas. Areas with a slope >= 40 degrees are processed with options -bulge 3.0 -ultra_fine, and areas with slope >= 60 are processed with -nature -ultra_fine. In addition, point clouds are rotated 8 times in 45 degree angles, tilted each time (by 20 and 30 degrees, respectively) and then reclassified.",
        "Mask layer for cloud regions. At the end of processing, if any points remain within 1 m of the upper limit of canopy height (typically 125 m), they and the immediate area surrounding them are masked out by this layer (set to NA). This will also mask dense high noise that automatic noise filters have missed, and DTM artefacts around inaccurately rendered cliffs. This mask is included in mask_combined.tif and should always be applied.",
        "Mask layer that combines several other masks, including: mask_cloud.tif, mask_noground.tif, mask_pd02.tif. This is a minimum mask to improve robustness of the inferred terrain and canopy models. Consider combining this layer with mask_steep.tif or mask_unstabledtm.tif for maximum robustness.",
        "Mask layer for regions of the point cloud devoid of ground points. Any areas that do not have any ground points within a circle of radius 15 m are masked out by this layer (set to NA). Can indicate areas of poor sampling such as in low-quality point clouds of dense forest canopies, or inaccurate DTM modelling, e.g. mountain spurs that have been cut off.  This mask is included in mask_combined.tif and should always be applied.",
        "Mask layer for regions of the point cloud where the DTM varies depending on which algorithm is used. Computed as areas where dtm_lasdef.tif and dtm_lasfine.tif differ by more than 2 m  after smoothing them with a circle of 10 m radius. Can be used to improve robustness, but may discard perfectly valid high-elevation or rugged regions.",
        "Mask layer for regions of the point cloud with very low pulse density. Masks out all areas with pulse densities < 2 pulses per m2 (set to NA). This mask is included in mask_combined.tif and should always be applied. Cf. recommendations in: https://doi.org/10.1111/2041-210X.14416, in particular Fig. 4.",
        "Mask layer for regions of the point cloud with low pulse density. Masks out all areas with pulse densities < 4 pulses per m2 (set to NA). Can be used to improve robustness of inference in dense canopies, but may discard perfectly valid regions. Cf. recommendations in: https://doi.org/10.1111/2041-210X.14416, in particular Fig. 4.",
        "Mask layer for steep regions of the point cloud where the DTM varies depending on which algorithm is used. Computed as areas where dtm.tif and dtm_lasdef.tif differ by more than 0.5 m after smoothing them with a circle of 10 m radius. Can be used to improve robustness, but may discard perfectly valid high-elevation or rugged regions.",
        "Outline of the airborne laser scan, in the local coordinate reference system. Includes the same information as summary_processing.csv. Useful to obtain metadata and processing details, but also to create buffers around raster products. Details on the information provided can be found in _INFO_summary_processing.csv.",
        "Outline of the airborne laser scan, in the local coordinate reference system. Includes the same information as summary_processing.csv. Useful to obtain metadata and processing details, and to combined multiple outlines for an overview of scan locations. Details on the information provided can be found in _INFO_summary_processing.csv.",
        "Pulse density raster based on last returns. Useful for quality checks and as predictor in models. Cf. recommendations in: https://doi.org/10.1111/2041-210X.14416, in particular Fig. 4.",
        "Pulse density raster based on last returns, for laser pulses with absolute scan angles <= 20 degrees. Useful to find high quality regions based on scanangle.",
        "Pulse density raster based on last returns. The default pulse density product. Useful for quality checks and as predictor in models. Cf. recommendations in: https://doi.org/10.1111/2041-210X.14416, in particular Fig. 4.",
        "Absolute scan angle raster. Average scan angle of points registered in each grid cell. Useful to check flight patterns and assess scan quality.",
        "Metadata and processing summary statistics. Useful for quick assessments of scans. Details on the information provided can be found in _INFO_summary_processing.csv."
      )
    )
    
    fwrite(products, file = file_output, col.names = T)
    # cat(name_file,"created at:", path_output,"\n")
  }
}


make.info_summary_processing = function(path_output = "", name_file = "_INFO_summary_processing.csv"){
  
  if(path_output != ""){
    # make output file
    file_output = file.path(path_output, name_file)
    # Create a data table with file names and descriptions
    summaries = data.table(
      column = c(
        "dataset",
        "site",
        "source",
        "citation",
        "contact",
        "access",
        "license",
        "updated",
        "id_orig",
        "acq_type",
        "acq_crs",
        "acq_units",
        "acq_mindt",
        "acq_maxdt",
        "acq_system",
        "acq_AGL",
        "acq_swath",
        "acq_wavel",
        "acq_fpsize",
        "acq_lasdiv",
        "notes",
        "issues",
        "height_lim",
        "angle_lim",
        "class_rm",
        "exclass_rm",
        "prms_lspkf",
        "crs",
        "lon",
        "lat",
        "area_km2",
        "GPSadjstd",
        "mingps",
        "maxgps",
        "mindt",
        "maxdt",
        "pd_point",
        "pd_pulse",
        "sdpd_pulse",
        "frac_grnd",
        "angle99th",
        "perc_pd02",
        "perc_pd04",
        "perc_steep",
        "perc_nogrd",
        "perc_undtm",
        "perc_cloud",
        "elev",
        "elev_min",
        "elev_max",
        "chm_mean",
        "chm_sd",
        "chm_perc99",
        "chm_max",
        "cc2",
        "cc10",
        "type_os",
        "type_arch",
        "v_lastools",
        "dir_input",
        "dir_output",
        "time_start",
        "time_end",
        "mins_total",
        "mins_grdcl",
        "mins_grdrf",
        "mins_lspkf",
        "type_file",
        "n_files",
        "n_cores",
        "adjacents",
        "size_MBin",
        "size_MBout"
      ),
      description = c(
        "Informal dataset descriptor",
        "Acquisition site. Empty if several sites are provided or no site name is known",
        "Data contributors. GCA01 if data have been downloaded from open repository",
        "Reference of relevant article or a link to the data source",
        "Email address of contributor(s). Empty if data are openly accessible",
        "Details on data accessibility (open vs. closed)",
        "If applicable, license type (eg., CC-BY 4.0)",
        "Date when metadata were last updated. In dd/mm/YYYY format",
        "If applicable, original identifier in source database",
        "Type of scan (ALS, DLS)",
        "Coordinate reference system of raw data. Usually supplied as EPSG code (e.g., EPSG:32650)",
        "Units in which raw data have been supplied (m, ft). Can be both m and ft if horizontal and vertical units differ.",
        "First date of acquisition as found in flight information. If not known exactly, first day of month when acquisition was started. In dd/mm/YYYY format",
        "Last date of acquisition as found in flight information. If not known exactly, last day of month when acquisition was started. In dd/mm/YYYY format",
        "Laser scanning instrument",
        "Flight altitude above ground in m",
        "Field of view or entire swath of scan in degrees. May not always be reliable due to different reporting conventions",
        "Wavelength in nm",
        "Footprint diameter in cm",
        "Laser divergence in mrad",
        "Any additional information",
        "Problems noted either in processing or beforehand (cloud, noise, etc.)",
        "Processing instruction. Height limit in m above ground above which points are automatically removed. Defaults to 125 m",
        "Processing instruction. Scan angle in degrees above which points are automatically removed. Defaults to NA (no removal)",
        "Processing instruction. Points with these classifications are automatically removed during processing. Defaults to NA (no removal)",
        "Processing instruction. Points with these extended classifications are automatically removed during processing. Defaults to NA (no removal)",
        "Parameters used to create the locally adaptive spikefree DSM and CHM (dsm_lspikefree.tif and chm_lspikefree.tif)",
        "Coordinate reference system in which products are delivered. Usually same as acq_crs, but overwritten by conflicting information in LAS files and, if necessary, re-projected to metric systems",
        "Longitude of centroid",
        "Latitude of centroid",
        "Total area of acquisition in km2",
        "Flags if GPS timestamps are in adjusted standard time. This is a best guess based on flags in LAS header and actual extents of timestamps",
        "Minimum GPS timestamp across LAS files",
        "Maximum GPS timestamp across LAS files",
        "Consensus estimate of first date of acquisition. Relies on GPS timestamps from LAS files if in adjusted standard time (dropping any timestamps that are set to 0 or large than current date), otherwise set to acq_mindt",
        "Consensus estimate of last date of acquisition. Relies on GPS timestamps from LAS files if in adjusted standard time (dropping any timestamps that are set to 0 or large than current date), otherwise set to acq_maxdt",
        "Mean point density (m-2)",
        "Mean pulse density (m-2)",
        "Standard deviation of pulse density (m-2)",
        "Fraction of ground points (unitless)",
        "Maximum scan angle of acquisition (99th percentile)",
        "Percentage of area with pulse density < 2 m-2",
        "Percentage of area with pulse density < 4 m-2",
        "Percentage of area with steep areas",
        "Percentage of area without ground points",
        "Percentage of area with potentially unstable DTMs",
        "Percentage of area with potential cloud cover",
        "Mean elevation (m)",
        "Minimum elevation (m)",
        "Maximum elevation (m)",
        "Mean canopy height based on chm_highest.tif (m)",
        "Standard deviation of canopy height based on chm_highest.tif (m)",
        "The 99th percentile of canopy height based on chm_highest.tif (m)",
        "Maximum canopy height based on chm_highest.tif (m)",
        "Canopy cover at 2 m (fraction, unitless)",
        "Canopy cover at 10 m (fraction, unitless)",
        "Type of operating system (Windows, Linux, Darwin/macOS)",
        "Type of architecture (64 or 32)",
        "LAStools version",
        "Directory of raw data during processing. May correspond to actual location of data, but cannot generally be used as reference to raw data as folders may have been moved or processed elsewhere",
        "Directory of processed data during processing. May correspond to actual location of products, but cannot generally be used as reference to products as folders may have been moved or processed elsewhere",
        "Start time of processing",
        "End time of processing",
        "Minutes spent on processing of this geo-unit",
        "Minutes spent on initial ground classification",
        "Minutes spent on refinement of ground classification",
        "Minutes spent on creation of chm_lspikefree.tif",
        "File type that was processed (.las or .laz)",
        "Number of raw data files",
        "Number of cores used for processing",
        "Flags if adjacent folders been scanned for overlapping point clouds. This is used for tile-based processing workflows where buffers from adjacent tiles can be added without requiring to process all tiles togethers",
        "Total size of input dataset",
        "Total size of output layers"
      )
    )
    
    fwrite(summaries, file = file_output, col.names = T)
    # cat(name_file,"created at:", path_output,"\n")
  }
}

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
#### 1. Small helper functions ####
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
# these are small functions that help with general processing

check.path_lastools = function(path_lastools, logfile){
  has.path_lastools = F
  files_las2dem = list.files(path_lastools, pattern = "las2dem")
  
  if(any(files_las2dem == "las2dem") | any(files_las2dem == "las2dem64") | any(files_las2dem == "las2dem.exe") | any(files_las2dem == "las2dem64.exe") | any(files_las2dem == "las2dem.o") | any(files_las2dem == "las2dem64")) {
    has.path_lastools = T
  } else {
    # cat("Error. LAStools directory not found. Make sure to use the LAStools bin directory as path variable.\n") 
    # if(file.exists(logfile)){
    #   logfile_dt = data.table(path = as.character(NA), issue = paste0("ERROR! LAStools directory not found. Make sure to use the LAStools bin directory as path variable."))
    #   fwrite(logfile_dt, file = logfile, append = T)
    # }
  } 
  return(has.path_lastools)
}

# check user input
check.license = function(path_lastools, logfile){
  if(!file.exists(file.path(path_lastools,"lastoolslicense.txt"))){
    # Test existence of LAStools license
    cat("WARNING! No LAStools license found\n\n")
    cat("If you use LAStools for education or evaluation purposes, you can continue to use this tool, but will need to adjust the retiling size to reduce total point counts per file (typically < 1,000,000 points). For commercial or government purposes, a full license is needed. For more information email 'info@rapidlasso.de' (program execution will pause for 10 seconds):\n\n")
    if(file.exists(logfile)){
      logfile_dt = data.table(path = as.character(NA), issue = paste0("WARNING! No LAStools license found."))
      fwrite(logfile_dt, file = logfile, append = T)
    }
    pause.xseconds(10)
  }
}

check.type_os = function(type_os, logfile){
  # we wrap this check in tryCatch / we don't want the processing pipeline to stop working just because of a failed check
  status = tryCatch(
    {
      type_os = tolower(type_os)
      type_os = paste0(toupper(substr(type_os, 1, 1)), tolower(substr(type_os, 2, nchar(type_os))))
      
      type_os_automatic = Sys.info()["sysname"]
      
      if(type_os == "Auto" | type_os == "Automatic"){
        type_os = type_os_automatic
      } else if(type_os != type_os_automatic){
        cat("WARNING! Automatically detected system (",type_os_automatic,") does not correspond to manually provided system (", type_os,"). This may break the pipeline. Consider setting type_os as 'automatic' or correct manually\n")
        if(file.exists(logfile)){
          logfile_dt = data.table(path = as.character(NA), issue = paste0("WARNING! Automatically detected system (",type_os_automatic,") does not correspond to manually provided system (", type_os,"). This may break the pipeline. Consider setting type_os as 'automatic' or correct manually."))
          fwrite(logfile_dt, file = logfile, append = T)
        }
        pause.xseconds(10)
      } 
      
      if(!type_os %in% c("Windows","Linux")){
        cat("WARNING! Operating system needs to be one of 'Linux' or 'Windows'. Other types, such as ",os_type, ", are not supported by LAStools.\n")
        if(file.exists(logfile)){
          logfile_dt = data.table(path = as.character(NA), issue = paste0("WARNING! Operating system needs to be one of 'Linux' or 'Windows'. Other types, such as ",os_type, ", are not supported by LAStools."))
          fwrite(logfile_dt, file = logfile, append = T)
        }
        pause.xseconds(10)
      } 
    }
    , error = function(e){
      return("error")
    }
  )
  cat("System:",type_os,"\n")
  return(type_os)
}

check.type_architecture = function(type_architecture, type_os, logfile){
  # we wrap this check in tryCatch / we don't want the processing pipeline to stop working just because of a failed check
  status = tryCatch(
    {
      type_architecture_automatic = as.character(Sys.info()["machine"])
      if(type_architecture_automatic %in% c("x86_64","x86-64","arm64","aarch64","ppc64","ppc64le","sparc64","s390x")){
        type_architecture_automatic = "64"
      } 
      
      if(type_architecture == "automatic"){
        type_architecture = type_architecture_automatic
        cat("Detected a ",type_architecture, " architecture\n")
      }
      
      if(type_architecture != "64"){type_architecture = "32"}
      
      if(type_architecture != "64" & type_os != "Windows"){
        cat("WARNING! LAStools currently only provides 64bit support for non-Windows systems. Consider setting type_architecture to 'automatic' or correct manually\n")
        if(file.exists(logfile)){
          logfile_dt = data.table(path = as.character(NA), issue = paste0("WARNING! LAStools currently only provides 64bit support for non-Windows systems. Consider setting type_architecture to 'automatic' or correct manually"))
          fwrite(logfile_dt, file = logfile, append = T)
        }
        pause.xseconds(10)
      } 
    }
    , error = function(e){
      return("error")
    }
  )
  cat("Architecture:",type_architecture,"bit\n")
  return(type_architecture)
}

check.n_cores = function(n_cores, type_os, logfile){
  # we wrap this check in tryCatch / we don't want the processing pipeline to stop working just because of a failed check
  status = tryCatch(
    {
      n_cores_automatic = 1
      if(type_os == "Windows"){
        n_cores_automatic = system("wmic cpu get NumberOfCores", intern = TRUE)
      } else if(type_os == "Linux"){
        n_cores_automatic = system("lscpu | grep 'Socket(s):' | awk '{print $2}'", intern = TRUE)
      } else if(type_os == "Darwin" | type_os == "macOS"){
        n_cores_automatic = system("sysctl -n hw.physicalcpu", intern = TRUE) 
      }
      n_cores_automatic = as.numeric(n_cores_automatic)
      
      if(n_cores == "auto" | n_cores == "automatic"){
        n_cores = max(1, n_cores_automatic - 2)
      } 
      
      n_cores = as.numeric(n_cores)
      
      if(n_cores > n_cores_automatic){
        cat("WARNING! It seems that the number of provided cores (", n_cores, ") is greater than the number of physical cores (", n_cores_automatic ,"). This may break the pipeline. Consider reducing the number of provided cores or setting n_cores to 'automatic'. Ignore this message if you know that there are enough physical cores.\n")
        if(file.exists(logfile)){
          logfile_dt = data.table(path = as.character(NA), issue = paste0("WARNING! It seems that the number of provides cores (", n_cores, ") is greater than the number of physical cores (", n_cores_automatic ,"). This may break the pipeline. Consider reducing the number of provided cores or setting n_cores to 'automatic'. Ignore this message if you know that there are enough physical cores."))
          fwrite(logfile_dt, file = logfile, append = T)
        }
      }
    }
    , error = function(e){
      return("error")
    }
  )
  cat("Cores:",n_cores,"\n")
  return(n_cores)
}

# update.blast2dem = function(use.blast2dem, type_os){
#   if(type_os != "Windows"){
#     # Linux currently does not support BLAST extension
#     use.blast2dem = F
#   }
#   cat("blast2dem is",ifelse(use.blast2dem == T,"used","not used"),"\n")
#   return(use.blast2dem)
# }

update.command_lastools = function(command_lastools, type_architecture){
  command_lastools = ifelse(type_architecture == "64" & !command_lastools %like% "64",paste0(command_lastools,"64"),command_lastools)
  return(command_lastools)
}

# update.command_lastools = function(command_lastools, type_architecture){
#   command_lastools = ifelse(type_architecture == "64" & !command_lastools %like% "64" & !command_lastools %like% "blast2dem",paste0(command_lastools,"64"),command_lastools)
#   return(command_lastools)
# }

# check whether a string is in any way non-valid (empty, null, na, etc.)
is.voidstring = function(x){
  if(!is.character(x) | is.null(x) | is.na(x) | is.nan(x) | trimws(x) == "") return(TRUE)
  else return(FALSE)
}

# make sure not read in files without any data
list.files.nonzero = function(...){
  files_read = list.files(...)
  files_sizes = unlist(lapply(files_read, file.size))
  files_read = files_read[files_sizes > 0]
  return(files_read)
}

# function to move files and unzip the folder if necessary
copy.files.totmp = function(path_input, path_tmp, type_file = ""){
  # recursive = TRUE, so that everything is removed
  files_toremove = list.files(path_tmp, include.dirs = F, full.names = T, recursive = T)
  unlink(files_toremove, recursive = TRUE)
  dirs_toremove = list.dirs(path_tmp, full.names = T, recursive = T)
  unlink(dirs_toremove[-1], recursive = TRUE)
  
  # now copy files from input directory into the temporary space
  # check whether files are zipped or not
  if(grepl(pattern = ".zip",x = path_input, fixed = TRUE)){
    # determine the file that needs copying
    file_tocopy = path_input
    
    cat("ZIP FILE: Copying zip file to temporary directory. This will take some time\n")
    file.copy(file_tocopy,to = path_tmp,overwrite = TRUE)
    
    cat("ZIP FILE: Unzipping. This will take some time\n")
    file_copied = list.files(path_tmp, full.names = T)
    unzip(file_copied, exdir = path_tmp)
    
    # removing zip file
    file.remove(file_copied)
  } else if(grepl(pattern = paste0(".",type_file),x = path_input, fixed = TRUE)  & file_test("-f",path_input)){ # file test may be overkill here
    file_tocopy = path_input
    file.copy(file_tocopy,to = path_tmp,overwrite = TRUE)
  } else {
    # simple copy operation
    cat("Copying files\n")
    files_tocopy = NULL
    if(type_file != ""){
      files_tocopy = list.files(path_input, pattern = paste0(".",type_file), full.names = T,recursive = T)
    } else {
      files_tocopy = list.files(path_input, full.names = T,recursive = T)
    }
    files_tocopy = files_tocopy[!like(files_tocopy, ".lasx",fixed = T) & !like(files_tocopy,".laszip",fixed = T)]
    file.copy(files_tocopy,to = path_tmp,overwrite = TRUE)
  }
}

# cleanup function
cleanup.files = function(path_data){
  files = list.files(path = path_data, full.names = T, pattern = "*[.]")
  files.removed = file.remove(files)
}

# adjust filepaths for utilization on Windows
# lastools function go through the shell, so there may be some issues with file path conventions on Windows. To address this, all file paths will be created via the R function file.path and then surrounded by quotation marks for the system call
file.path.system = function(type_os, ...){
  fp = file.path(...)
  if(type_os == "Windows"){fp = paste0("\"",fp,"\"")}
  return(fp)
}

# Get the scan angle field name to avoid conflict between las or laz point formats.
get.scan_angle_field = function(path_data, type_file = "laz") {
  
  # Read one file to inspect available fields
  files = list.files(path_data, pattern = paste0("\\.", type_file, "$"), 
                     full.names = TRUE)
  
  if (length(files) == 0) {
    warning("No files found, defaulting to 'ScanAngle'")
    return("ScanAngle")
  }
  
  # Read minimal data from first file
  las = readLAS(files[1], select = "a")  # 'a' = scan angle field
  available_fields = names(las@data)
  
  if ("ScanAngle" %in% available_fields) {
    cat("Scan angle field detected: ScanAngle\n")
    return("ScanAngle")
  } else if ("ScanAngleRank" %in% available_fields) {
    cat("Scan angle field detected: ScanAngleRank\n")
    return("ScanAngleRank")
  } else {
    warning("No scan angle field found, defaulting to 'ScanAngle'")
    return("ScanAngle")
  }
}

# based on terra package, get the max value of a raster and the value above which there are at least 100 points
# in some cases (virtual rasters, etc.), the min and max values have to be set/computed first
get.max_raster = function(file_raster){
  r = rast(file_raster)
  r = as.data.table(r)
  colnames(r) = c("z")
  setorder(r,-z)
  max_raster = max(r$z)
  first100 = min(r[1:100]$z)
  return(data.table(file_raster = file_raster, max_raster = max_raster, first100 = first100))
}

# function that automatically extracts coordinate reference from path
# this can be used to automatically provide coordinate systems for lidar files whose coordinate reference system is not set properly/cannot be extracted
# two options: provide a) a UTM string, in the format: "UTM30S", or b) an EPSG or ESRI code in the format: "EPSG4326" or "ESRI54009". Both upper and lower case are allowed, as well as underscores, e.g. "UTM_30S" or "EPSG_4326" / "ESRI_54009". For UTM codes, "UTM1S" is also accepted, but "UTM01S" should be used. Any other patterns will be ignored.

get.crs_from_path = function(path_file){
  
  # prepare extraction
  path_file = tolower(path_file)
  crs_from_path = NA
  
  # try utm
  match_utm = regexpr("utm_?[0-9]?[0-9](s|n)", path_file)
  
  if(match_utm >= 0){
    # we get position and length, and then analyze the last 3 letters of the utm substring (zone + hemisphere)
    match_position = as.integer(match_utm)
    match_length = attr(match_utm,"match.length")
    zone_utm = as.numeric(gsub("[^0-9.-]", "",substr(path_file, match_position+match_length-3, match_position+match_length-2)))
    hemisphere = substr(path_file, match_position+match_length-1, match_position+match_length-1)
    crs_from_path = paste0("EPSG:",ifelse(hemisphere == "n",326,327),sprintf("%02d", zone_utm))
  } else {
    # try ESRI and EPSG codes
    match_spatialref = regexpr("(epsg|esri)_?[0-9]+", path_file)
    if(match_spatialref >= 0){
      match_position = as.integer(match_spatialref)
      match_length = attr(match_spatialref,"match.length")
      crs_from_path = paste0(toupper(substr(path_file,match_position, match_position+3)),":",gsub("_","",substr(path_file, match_position+4,match_position+match_length-1)))
    }
  }
  
  return(crs_from_path)
}

get.dates_from_path = function(path_file){
  
  # prepare extraction
  path_file = tolower(path_file)
  dates_from_path = data.table(mindt = as.Date(NULL), maxdt = as.Date(NULL))
  
  # try conventional date format
  minmatch_dmY_nosep = regexpr(pattern = "mindt[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]", path_file)
  maxmatch_dmY_nosep = regexpr(pattern = "maxdt[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]", path_file)
  
  minmatch_dmY = regexpr(pattern = "mindt[0-9]?[0-9](_|\\-|\\.)[0-9]?[0-9](_|\\-|\\.)([0-9][0-9][0-9][0-9])", path_file)
  maxmatch_dmY = regexpr(pattern = "maxdt[0-9]?[0-9](_|\\-|\\.)[0-9]?[0-9](_|\\-|\\.)([0-9][0-9][0-9][0-9])", path_file)
  
  minmatch_dY = regexpr(pattern = "mindt[0-9]?[0-9]?[0-9](_|\\-|\\.)[0-9][0-9][0-9][0-9]", path_file)
  maxmatch_dY = regexpr(pattern = "maxdt[0-9]?[0-9]?[0-9](_|\\-|\\.)[0-9][0-9][0-9][0-9]", path_file)
  
  if(minmatch_dmY_nosep >= 0 & maxmatch_dmY_nosep >= 0){
    # we get position and length, and then analyze the last 3 letters of the utm substring (zone + hemisphere)
    minmatch_position = as.integer(minmatch_dmY_nosep)
    minmatch_length = attr(minmatch_dmY_nosep,"match.length")
    mindt_dmY = substr(path_file, minmatch_position+5, minmatch_position + minmatch_length-1)
    mindt_dmY = as.Date(mindt_dmY, "%d%m%Y")
    
    maxmatch_position = as.integer(maxmatch_dmY_nosep)
    maxmatch_length = attr(maxmatch_dmY_nosep,"match.length")
    maxdt_dmY = substr(path_file, maxmatch_position+5, maxmatch_position + maxmatch_length-1)
    maxdt_dmY = as.Date(maxdt_dmY, "%d%m%Y")
    
    # update dates_from_path
    dates_from_path = rbind(dates_from_path, data.table(mindt = mindt_dmY, maxdt = maxdt_dmY))
  } else if(minmatch_dmY >= 0 & maxmatch_dmY >= 0){
    minmatch_position = as.integer(minmatch_dmY)
    minmatch_length = attr(minmatch_dmY,"match.length")
    mindt_dmY = substr(path_file, minmatch_position+5, minmatch_position + minmatch_length-1)
    mindt_dmY = gsub(pattern = "(_|\\-|\\.)","/",mindt_dmY)
    mindt_dmY = as.Date(mindt_dmY, "%d/%m/%Y")
    
    maxmatch_position = as.integer(maxmatch_dmY)
    maxmatch_length = attr(maxmatch_dmY,"match.length")
    maxdt_dmY = substr(path_file, maxmatch_position+5, maxmatch_position + maxmatch_length-1)
    maxdt_dmY = gsub(pattern = "(_|\\-|\\.)","/",maxdt_dmY)
    maxdt_dmY = as.Date(maxdt_dmY, "%d/%m/%Y")
    
    # update dates_from_path
    dates_from_path = rbind(dates_from_path, data.table(mindt = mindt_dmY, maxdt = maxdt_dmY))
  } else if(minmatch_dY >= 0 & maxmatch_dY >= 0){
    minmatch_position = as.integer(minmatch_dY)
    minmatch_length = attr(minmatch_dY,"match.length")
    mindt_dmY = substr(path_file, minmatch_position+5, minmatch_position + minmatch_length-1)
    mindt_dmY = gsub(pattern = "(_|\\-|\\.)","/",mindt_dmY)
    mindt_dmY = as.Date(as.integer(tstrsplit(mindt_dmY,split = "/",fixed = T)[[1]])-1, origin = as.Date(paste0("01/01/",tstrsplit(mindt_dmY,split = "/",fixed = T)[[2]]), "%d/%m/%Y"))
    
    maxmatch_position = as.integer(maxmatch_dY)
    maxmatch_length = attr(maxmatch_dY,"match.length")
    maxdt_dmY = substr(path_file, maxmatch_position+5, maxmatch_position + maxmatch_length-1)
    maxdt_dmY = gsub(pattern = "(_|\\-|\\.)","/",maxdt_dmY)
    maxdt_dmY = as.Date(as.integer(tstrsplit(maxdt_dmY,split = "/",fixed = T)[[1]])-1, origin = as.Date(paste0("01/01/",tstrsplit(maxdt_dmY,split = "/",fixed = T)[[2]]), "%d/%m/%Y"))
    
    # update dates_from_path
    dates_from_path = rbind(dates_from_path, data.table(mindt = mindt_dmY, maxdt = maxdt_dmY))
  } else {
    dates_from_path = rbind(dates_from_path, data.table(mindt = as.Date(NA), maxdt = as.Date(NA)))
  }
  return(dates_from_path)
}

# https://stackoverflow.com/questions/1174799/how-to-make-execution-pause-sleep-wait-for-x-seconds-in-r
pause.xseconds = function(x)
{
  # p1 = proc.time()
  Sys.sleep(x)
  # proc.time() - p1 # The cpu usage should be negligible
}

# there is a slightly odd behavior with terra::buffer, where on some systems, Rstudio aborts with a fatal error when using it
# this likely has something to do with improper crs or no setting of crs, but I have not yet managed to systematically reproduce it across all systems (Windows Server, Windows 10, etc.) or scans
# it does not happen with st_buffer, so we create a small wrapper for now
buffer_withsf = function(vect_input, width){
  vect_output = vect(st_buffer(st_as_sf(vect_input), width))
  return(vect_output)
}

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
#### 2. lidR encapsulation ####
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
# helper functions that are only based on the lidR package
get.extent_lidar = function(path_input, type_file, attributes_vect = NULL, attributes_rast = NULL){
  
  # create a data.table that lists all las or laz files (their paths, names, and folders that they are in)
  files_toprocess = list.files(path_input, recursive = T, full.names = T, pattern = paste0(".",type_file))
  
  ctg_lidar = readLAScatalog(files_toprocess)
  extent_lidar = st_bbox(ctg_lidar)
  extent_lidar = as.polygons(ext(c(extent_lidar[1],extent_lidar[3],extent_lidar[2],extent_lidar[4])))
  terra::crs(extent_lidar) = st_crs(ctg_lidar)$input
  
  if(is.na(terra::crs(extent_lidar)) | is.null(terra::crs(extent_lidar)) | terra::crs(extent_lidar) == "") terra::crs(extent_lidar) = get.crs_from_path(path_input)
  if(!(is.na(terra::crs(extent_lidar)) | is.null(terra::crs(extent_lidar)) | terra::crs(extent_lidar) == "")){
    extent_lidar$path = path_input
    extent_lidar$area = st_area(ctg_lidar)
    extent_lidar$npoints = ctg_lidar@data$Number.of.point.records
    extent_lidar$npulse = ctg_lidar@data$Number.of.1st.return
    extent_lidar$nfile = dim(ctg_lidar@data)[1]
    
    extent_lidar$crs_original = terra::crs(extent_lidar)
    extent_lidar$crs_original_name = st_crs(ctg_lidar)$Name
    extent_lidar = project(extent_lidar,"EPSG:4326")
    
    if(!is.null(attributes_vect)){
      extent_lidar_centroid = centroids(extent_lidar)
      extent_lidar_centroid = project(extent_lidar_centroid, attributes_vect)
      extent_lidar = cbind(extent_lidar, extract(attributes_vect, extent_lidar_centroid)[,-1])
    }
    
    if(!is.null(attributes_rast)){
      extent_lidar_centroid = centroids(extent_lidar)
      extent_lidar_centroid = project(extent_lidar_centroid, attributes_rast)
      extent_lidar = cbind(extent_lidar, extract(attributes_rast, extent_lidar_centroid)[,-1])
    }
    
    return(extent_lidar)
  } else {
    return(NULL)
  }
}

# lidR equivalent of lasindex
# laxindex does not seem to explicitly exist anymore in lidR, we insert it manually
make.catalog_laxindex = function(ctg){
  
  stopifnot(is(ctg, "LAScatalog"))
  
  opt_chunk_size(ctg)   = 0
  opt_chunk_buffer(ctg) = 0
  opt_wall_to_wall(ctg) = FALSE
  
  create.file_lax = function(cluster) {
    rlas::writelax(cluster@files)
    return(0)
  }
  
  options = list(need_buffer = FALSE, drop_null = FALSE)
  
  catalog_apply(ctg, create.file_lax,.options = options)
  return(invisible())
}

ctg_summarize_chunk <- function(las_chunk, ...) {
  
  # Force reading full LAS file, not the cluster
  source_file <- las_chunk@files
  las <- readLAS(source_file)
  
  if (lidR::is.empty(las)) return(NULL)
  
  las.ground <- filter_ground(las)
  
  points <- nrow(las@data)
  points_ground <- nrow(las.ground@data)
  fraction_ground <- points_ground / points
  
  area <- as.numeric(sf::st_area(las))
  
  mingps <- min(las$gpstime, na.rm = TRUE)
  maxgps <- max(las$gpstime, na.rm = TRUE)
  
  pulses <- las@header[["Number of points by return"]][1]
  
  density_points <- if (area > 0) points / area else NA_real_
  density_pulses <- if (area > 0) pulses / area else NA_real_
  spacing_pulses <- if (density_pulses > 0) sqrt(1 / density_pulses) else NA_real_
  
  data.table(
    classified = points_ground > 0,
    points = points,
    points_ground = points_ground,
    fraction_ground = fraction_ground,
    density_points = density_points,
    density_pulses = density_pulses,
    spacing_pulses = spacing_pulses,
    area = area,
    mingps = mingps,
    maxgps = maxgps
  )
}


summarize.lidR = function(lasfile){
  
  las = readLAS(lasfile)
  
  if (lidR::is.empty(las)) return(NULL)
  
  las.ground = filter_ground(las)
  points = nrow(las@data)
  points_ground = nrow(las.ground@data)
  fraction_ground = points_ground/points
  area = as.numeric(st_area(las))
  
  mingps = min(las@data[,"gpstime"], na.rm=T)
  maxgps = max(las@data[,"gpstime"], na.rm=T)
  
  pulses = las@header[["Number of points by return"]][1]
  
  density_points = if(area > 0) points/area else as.numeric(NA)
  density_pulses = if(area > 0) pulses/area else as.numeric(NA)
  spacing_pulses = if(density_pulses > 0) sqrt(1/density_pulses) else as.numeric(NA)
  
  return(data.table(classified = ifelse(points_ground != 0, TRUE, FALSE), 
                    points = points, 
                    points_ground = points_ground, 
                    fraction_ground = fraction_ground, 
                    density_points = density_points, 
                    density_pulses = density_pulses, 
                    spacing_pulses = spacing_pulses, 
                    area = area, 
                    mingps = mingps, 
                    maxgps = maxgps))
}

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
#### 3. LAStools/lidR encapsulation ####
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#

# functions that are either based directly on LAStools or on lidR package functions
# when functions are directly based on LAStools functions, they will share the same name, otherwise a custom name will be given
# lastools functions are directly encapsulated via system call
# lidR equivalents are provided within the same functions and replace lastools functions as soon as the path_lastools variable is empty/NULL/NA
# all functions are given in approximate order in which they are used in the final script
# all functions take a params_general object and return it (potentially modified) to guarantee consistency across the pipeline

# get the version
get.version_lastools = function(params_general){
  if(!is.voidstring(params_general$path_lastools)){
    return_system = system(
      paste(
        file.path.system(type_os = params_general$type_os, params_general$path_lastools, update.command_lastools("las2las", type_architecture = params_general$type_architecture)),
        "-version",
        ifelse(params_general$type_os == "Linux","2>&1", "") # crucial for Linux where LAStools seems to write to stderr
      ), intern = T
    )
    
    if(length(return_system) > 0){
      return_system = tstrsplit(return_system, " ")
      version_lastools = return_system[length(return_system)]
      version_lastools = version_lastools[[1]]
      return(version_lastools)  
    } else {
      return(as.character(NA))
    }
  } else {
    return(as.character(NA))
  }
}

# generic lastools call function
call.lastools = function(command_lastools, arguments_lastools, params_general, type_output = "laz", path_output = "", update.path = TRUE, return.time = T){
  
  # define output directory
  if(is.voidstring(path_output)){
    path_output = file.path(params_general$path_data, command_lastools)
  }
  if(!dir.exists(path_output)) dir.create(path_output,recursive = T)
  
  command_lastools = update.command_lastools(command_lastools = command_lastools, type_architecture = params_general$type_architecture)
  
  # pass lastools command to system
  if(!is.voidstring(params_general$path_lastools)){
    return_system = system(
      paste(
        file.path.system(type_os = params_general$type_os, params_general$path_lastools, command_lastools),
        "-i", file.path.system(type_os = params_general$type_os, params_general$path_data,paste0("*.",params_general$type_file)),
        "-cores", params_general$n_cores,
        "-odir", file.path.system(type_os = params_general$type_os, path_output),
        paste0("-o",type_output),
        arguments_lastools,
        ifelse(params_general$type_os == "Linux","2>&1", "") # crucial for Linux where LAStools seems to write to stderr
      )
    )
    
    if(return_system == 0){
      cat("Successful completion of system command\n")
    } else {
      # logfile_dt = data.table(path = as.character(NA), issue = paste0("WARNING! Call to", command_lastools, "generated non-0 return."))
      # fwrite(logfile_dt, file = params_general$logfile, append = T)
      cat("System command could not be completed\n")
    }
  } else {
    cat("Did not find LAStools software\n")
  }
  
  if(params_general$cleanup == T) cleanup.files(params_general$path_data)
  
  params_general$path_data = path_output
  params_general$type_file = "laz"
  
  if(command_lastools == "lastile"){
    params_general$bbtype = "tile"
  }
  
  if(type_output %in% c("las","laz")){
    cat("Reindexing\n")
    # lasindex(params_general)
  }
  
  if(update.path == TRUE){
    cat("Amended file path returned\n")
    return(params_general)
  }
}


# create (spatial) indexing of las files to speed up processing
lasindex = function(params_general){
  
  # files_toindex = list.files(params_general$path_data, pattern = paste0(".",params_general$type_file))
  #
  # if(length(files_toindex) > 1){
  # lastools indexing tool is free/open, but might be handy to have lidR option on mac/linux systems
  
  if(!is.voidstring(params_general$path_lastools)){
    
    # pass lastools command to system
    return_system = system(
      paste(
        file.path.system(type_os = params_general$type_os, params_general$path_lastools, update.command_lastools("lasindex", type_architecture = params_general$type_architecture)),
        "-i",
        file.path.system(type_os = params_general$type_os, params_general$path_data,paste0("*.",params_general$type_file)),
        "-cores", params_general$n_cores,
        ifelse(params_general$type_os == "Linux","2>&1", "") # crucial for Linux where LAStools seems to write to stderr
      )
    )
    
    # check return
    if(return_system == 0){
      cat("Successful completion of system command\n")
    } else {
      # logfile_dt = data.table(path = as.character(NA), issue = paste0("WARNING! Lasindex generated non-0 return."))
      # fwrite(logfile_dt, file = params_general$logfile, append = T)
      cat("System command could not be completed\n")
    }
  } else {
    
    files.input = list.files(path = params_general$path_data,pattern = paste0("\\.",params_general$type_file), full.names = TRUE)
    ctg = readLAScatalog(files.input)
    
    opt_filter(ctg) = "-drop_withheld"
    opt_output_files(ctg) = params_general$path_data
    opt_progress(ctg) = T
    opt_laz_compression(ctg) = T
    make.catalog_laxindex(ctg)
  }
}

# try out
# lasindex(params_general)
# lasinfo to repair headers
lasinfo.repair =  function(params_general){
  # pass lastools command to system
  if(!is.voidstring(params_general$path_lastools)){
    return_system = system(
      paste(
        file.path.system(type_os = params_general$type_os, params_general$path_lastools, update.command_lastools("lasinfo", type_architecture = params_general$type_architecture)),
        "-i",
        file.path.system(type_os = params_general$type_os, params_general$path_data,paste0("*.",params_general$type_file)),
        "-cores", params_general$n_cores,
        "-repair",
        "-quiet",
        ifelse(params_general$type_os == "Linux","2>&1", "") # crucial for Linux where LAStools seems to write to stderr
      )
    )
  } else {
    # !!! TODO: lidR version
  }
}

# new in v.44, unifies cleaning and reprojection operations into a single operation (previously done in lastile and las2utm)
# new in v.44: this new operation adds additional features, such as the removal of isolated points outside of bounding box, and remove.vlr and remove.evlr. Sometimes problematic (E)VLRs cause LAStools to crash, so this should not be a problem anymore
# also new in v.44: option to rescale the lasfiles, e.g. by setting factor_rescale = 0.01, but this needs to be done carefully (only after inspection of files)
# initial cleaning
# can also be used to force to UTM
las2las.initial = function(params_general, metadata, size_tile, path_output = "", force.utm = F, remove.vlr = F, remove.evlr = F, factor_rescale = NULL, angle_lim = NULL, update.path = TRUE, class_rm = "", exclass_rm = "", type_point = NULL){
  
  # define output directory
  if(path_output == ""){
    path_output = file.path(params_general$path_data,"las2las")
  }
  if(!dir.exists(path_output)) dir.create(path_output)
  
  # # pass lastools command to system
  if(!is.voidstring(params_general$path_lastools)){
    return_system = system(
      paste(
        file.path.system(type_os = params_general$type_os, params_general$path_lastools, update.command_lastools("las2las", type_architecture = params_general$type_architecture)),
        "-i",
        file.path.system(type_os = params_general$type_os, params_general$path_data,paste0("*.",params_general$type_file)),
        "-cores", params_general$n_cores,
        "-drop_withheld",
        paste0("-drop_class 7 18 ", class_rm), # noise classes (noise / high noise) # new in v.47: remove extra noise classes or otherwise from scan (necessary for IGN France, for example, where artefacts/noise is usually marked with 65, or even 28)
        ifelse(nchar(exclass_rm) > 0, paste0("-drop_extended_class ", exclass_rm),""), # new in v.47: remove extra noise classes or otherwise from scan (necessary for IGN France, for example, where artefacts/noise is usually marked with 65, or even 28)
        "-crop_to_bounding_box", # remove xy outliers
        ifelse(remove.vlr == T,"-remove_all_vlrs",""),
        ifelse(remove.evlr == T,"-remove_all_evlrs",""),
        ifelse(force.utm == T,"-target_utm auto",""),
        ifelse(!is.null(factor_rescale),paste("-rescale",factor_rescale,factor_rescale,factor_rescale,sep = " "),""),
        ifelse(!is.null(angle_lim),paste("-drop_abs_scan_angle_above", angle_lim, sep = " "),""),
        ifelse(!is.null(type_point), paste0("-set_point_type ",type_point),""), # remove any RGB or other information
        "-odir", file.path.system(type_os = params_general$type_os, path_output),
        "-olaz",
        ifelse(params_general$type_os == "Linux","2>&1", "") # crucial for Linux where LAStools seems to write to stderr
      )
    )
    
    # lasindex(params_general)
  } else {
    
    cat("las2las.initial OpenSource version!\n")
    
    # drop noise, withheld, high returns
    # TODO:
    # - add metadata/fields filter
    # - keep withheld ?
    # - remove foreach
    
    angle_lim = NULL
    
    filter = c(ifelse(!is.null(angle_lim), paste("-drop_abs_scan_angle_above", angle_lim, sep = " "),""),
               "classification %out% 7 18",
               "-drop_withheld")
    
    output_filename <- file.path(path_output, "*.laz")
    
    epsg_code <- as.numeric(strsplit(metadata$acq_crs, split = ":")[[1]][2])
    
    clean_data_pipeline <-  lasR::reader(
      filter = filter
    ) + lasR::set_crs(epsg_code) + lasR::write_las(output_filename)
    
    files <- list.files(params_general$path_data, pattern = paste0("\\.", type_file), full.names = TRUE)
    
    lasR::exec(
      clean_data_pipeline,
      on = files,
      ncores = params_general$n_cores,
      progress = T
    )
    
    # cl <- makeCluster(params_general$n_cores)
    # registerDoParallel(cl)
    # 
    # 
    # foreach(filename = files, .combine = c) %dopar% {
    #   las2las_custom_function(filename, path_output, angle_lim, metadata, size_tile)
    # }
    # 
    # stopCluster(cl)
  }
  
  if(update.path == TRUE){
    
    if (params_general$cleanup == T) {
      cat("Removing copied files\n")
      copied_files = list.files(params_general$path_data, pattern=paste0("\\.", type_file), full.names = TRUE)
      file.remove(copied_files)}
    
    cat("Amending file path\n")
    params_general$path_data = path_output
    params_general$type_file = "laz"
    cat("Amended file path returned\n")
    return(params_general)
  }
  
}

# cluster of point cloud data
# reasoning:
# - sometimes, lidar flight lines/tiles do not overlap/touch; this can create separate point cloud segments
# - if segments are large or very far away from each other, it is much more efficient to process each separate segment separately
# - therefore, we need to determine which las files belong to the same segment of the point cloud and restructure the processing accordingly

# Method:
# - check for overlap or adjacency between individual flight lines and tiles (las files)
# - implement a 10 m buffer zone, i.e. if two flight lines / tiles do not have any points within 20m of each other, they should be assumed to be separate
# - sort all overlapping las files into clusters, and regroup accordingly, with a separate params_general for processing

# NOTES:
# in most use case, point clouds won't contain separate clusters, so this step will simply return the initial params_general
# if data sets are small or cover small extents, this step could be ignored, but running it never hurts, as it splits up any scan into sensible units, so it is a default
# nbclusters_forced = 10

find.clusters_data = function(params_general, path_output = "", nbclusters_forced = NULL){
  
  files_tocluster = list.files(params_general$path_data, pattern = paste0("\\.", params_general$type_file), full.names = T)
  
  # if(length(files_tocluster) > 1){
  # define output directory
  if(path_output == ""){
    path_output = file.path(params_general$path_data,"clusters")
  }
  if(!dir.exists(path_output)) dir.create(path_output)
  
  if(!is.voidstring(params_general$path_lastools) & file.exists(file.path(path_lastools,"lastoolslicense.txt"))){
    # create output directory for temporary processing
    path_tmp = file.path(params_general$path_data,"tmp")
    if(!dir.exists(path_tmp)) dir.create(path_tmp)
    
    # define the enclosing boundary
    # we are only interested in 2D boundaries and only need approximate boundaries
    # so everything is collapsed to z values of 0 via clamp_z 0 0 and thinned at m2 scale
    return_system = system(
      paste(
        file.path.system(type_os = params_general$type_os, params_general$path_lastools, update.command_lastools("lasthin", type_architecture = params_general$type_architecture)),
        "-i",
        file.path.system(type_os = params_general$type_os, params_general$path_data,paste0("*.",params_general$type_file)),
        "-cores", params_general$n_cores,
        "-odir", file.path.system(type_os = params_general$type_os, path_tmp),
        "-lowest",
        "-drop_withheld",
        "-step 1.0",
        "-clamp_z 0 0",
        "-olaz",
        ifelse(params_general$type_os == "Linux","2>&1", "") # crucial for Linux where LAStools seems to write to stderr
      )
    )
    
    # now define boundaries of each individual lasfile and write to shape file
    return_system = system(
      paste(
        file.path.system(type_os = params_general$type_os, params_general$path_lastools, update.command_lastools("lasboundary", type_architecture = params_general$type_architecture)),
        "-i",
        file.path.system(type_os = params_general$type_os, path_tmp,"*.laz"),
        "-cores", params_general$n_cores,
        ifelse(length(files_tocluster) > 1, "-use_bb",""), # makes it faster if it's just a single file where no overlap can occur
        "-odir", file.path.system(type_os = params_general$type_os, path_tmp),
        "-oshp",
        ifelse(params_general$type_os == "Linux","2>&1", "") # crucial for Linux where LAStools seems to write to stderr
      )
    )
    
    # now cluster them
    # we use igraph package to construct a graph of connected acquisitions and then extract isolated clusters
    outlines_files = list.files(path_tmp, pattern = ".shp", full.names = T)
    
    # read in generated shapefiles and connect them back to the las files
    outlines_input = lapply(outlines_files,function(x){y = vect(x);y$files = gsub(".shp",paste0(".",params_general$type_file),x, fixed = T);if(nrow(geom(y)) > 0){return(y)}else{return(NULL)}})
    outlines_input = outlines_input[which(!sapply(outlines_input, is.null))] # remove NULL elements
    
    outlines_input = vect(outlines_input)
    # outlines_input$files = list.files(params_general$path_data,pattern = params_general$type_file, full.names = T) # deprecated, need to check whether lastools output gave back proper geometry
    buffer_outlines = params_general$buffer
    outlines_input_buffered = buffer_withsf(outlines_input, buffer_outlines) # implicitly removes invalid outlines
    
    # cluster
    outlines_adjacency = relate(outlines_input_buffered, outlines_input_buffered,"intersects")
    outlines_graph = graph_from_adjacency_matrix(outlines_adjacency, diag = F)
    #plot(outlines_graph)
    
    names_outlines_clusters = c("membership","csize","no")
    outlines_clusters = vector("list", length(names_outlines_clusters))
    names(outlines_clusters) = names_outlines_clusters
    if(!is.null(nbclusters_forced)){
      if(nbclusters_forced >= 1 & nbclusters_forced <= length(outlines_files)){
        cat("Finding clusters of nearby point cloud files\n")
        clusters = cluster_fluid_communities(outlines_graph,round(nbclusters_forced))
        outlines_clusters$membership = clusters$membership
        outlines_clusters$csize = clusters$vcount
        outlines_clusters$no = length(unique(clusters$membership))
      } else {
        cat("Could not force number of clusters. Nb is either too small (< 1) or too large (more than available las/laz files)\n")
        cat("Finding clusters of disconnected point cloud files\n")
        outlines_clusters = components(outlines_graph)
      }
    } else {
      cat("Finding clusters of disconnected point cloud files\n")
      outlines_clusters = components(outlines_graph)
    }
    
    # update outlines and write to file
    outlines_input$cluster = outlines_clusters$membership
    outlines_input$buffer = buffer_outlines
    
    # update buffered outlines
    outlines_input_buffered$cluster = outlines_clusters$membership
    outlines_input_buffered$buffer = buffer_outlines
    # writeVector(outlines_input_buffered, filename = file.path(path_output,"outlines_input_buffered.shp"), overwrite = TRUE)
    
    # create params_general for each cluster and move data into subfolders
    nbclusters = outlines_clusters$no
    list_params_general = vector(mode = "list", length = nbclusters)
    
    if(nbclusters > 1){
      # first we update params_general and copy buffers of the individual clusters into subfolders
      for(i in 1:nbclusters){
        # update path
        params_general_cluster = data.table::copy(params_general)
        params_general_cluster$path_data = file.path(params_general$path_data,paste0("part_",i))
        
        # create new directory specifically for this cluster
        if(!dir.exists(params_general_cluster$path_data)) dir.create(params_general_cluster$path_data)
        list_params_general[[i]] = params_general_cluster
        
        # get polygon buffers
        outlines_buffer = aggregate(outlines_input_buffered[outlines_input_buffered$cluster == i])
        bufferfiles_current = file.path(params_general$path_data, basename(outlines_input[is.related(outlines_input, outlines_buffer,"intersects") & outlines_input$cluster != i]$files))
        
        if(length(bufferfiles_current) > 0){
          # write the extent
          path_outlines_buffer = file.path(params_general_cluster$path_data,"outlines_buffer.shp")
          writeVector(outlines_buffer, filename = path_outlines_buffer, overwrite = T)
          
          # now copy the files
          file.copy(bufferfiles_current, params_general_cluster$path_data)
          
          # copy buffer files (with clipping) into subdirectory
          cleanup_default = params_general_cluster$cleanup
          params_general_cluster$cleanup = F
          call.lastools(command_lastools = "lasclip",arguments_lastools = paste0("-drop_withheld -poly ",file.path.system(type_os = params_general$type_os, path_outlines_buffer)," -odix _clipped"),params_general = params_general_cluster, path_output = params_general_cluster$path_data, type_output = "laz")
          params_general_cluster$cleanup = cleanup_default
          
          # remove extraneous files
          files_toremove = list.files(params_general_cluster$path_data, full.names = T)
          files_toremove = files_toremove[!files_toremove %like% "_clipped.laz" & !files_toremove %like% "_clipped.las"]
          file.remove(files_toremove)
        }
      }
      
      # now move the files
      for(i in 1:nbclusters){
        params_general_cluster = list_params_general[[i]]
        files_current = file.path(params_general$path_data, basename(outlines_input[outlines_input$cluster == i]$files))
        files_current_new = file.path(params_general_cluster$path_data, basename(files_current))
        file.rename(files_current, files_current_new)
      }
    } else{
      list_params_general[[1]] = params_general
    }
    
    # write out the polygons
    writeVector(outlines_input, filename = file.path(params_general$path_data,"outlines_input.shp"), overwrite = TRUE)
    
    # clean up
    unlink(path_tmp,recursive = T)
    
    return(list_params_general)
  } else {
    
    cat("Open source clustering")
    
    # ctg = readALScatalog(files_tocluster)
    # clusters = lidR::catalog_split_clusters(ctg)
    # 
    # nbclusters = length(clusters)
    # list_params_general = vector(mode = "list", length = nbclusters)
    # 
    # if(nbclusters > 1){
    #   # first we update params_general and copy buffers of the individual clusters into subfolders
    #   for(i in 1:nbclusters){
    #     # update path
    #     params_general_cluster = data.table::copy(params_general)
    #     params_general_cluster$path_data = file.path(params_general$path_data,paste0("part_",i))
    #     
    #     # create new directory specifically for this cluster
    #     if(!dir.exists(params_general_cluster$path_data)) dir.create(params_general_cluster$path_data)
    #     list_params_general[[i]] = params_general_cluster
    #     
    #     # get polygon buffers
    #     outlines_buffer = aggregate(outlines_input_buffered[outlines_input_buffered$cluster == i])
    #     bufferfiles_current = file.path(params_general$path_data, basename(outlines_input[is.related(outlines_input, outlines_buffer,"intersects") & outlines_input$cluster != i]$files))
    #     
    # # create output directory for temporary processing
    # path_tmp = file.path(params_general$path_data,"tmp")
    # if(!dir.exists(path_tmp)) dir.create(path_tmp)
    # 
    ctg = readLAScatalog(files_tocluster)
    outlines_input = vect(ctg@data)
    outlines_input = outlines_input[,c("filename")]
    outlines_input$cluster = 1
    buffer_outlines = params_general$buffer
    outlines_input$buffer = buffer_outlines
    writeVector(outlines_input, filename = file.path(params_general$path_data,"outlines_input.shp"), overwrite = TRUE)
    
    list_params_general = list(params_general)
    return(list_params_general)
  }
}

# find adjacent tiles from other folders in the same directory
add.adjacent = function(outlines_cluster, path_origin, path_moveto, type_file, buffer){
  
  tmpfile_files_extent = list.files(tempdir(), pattern = "tmpfile_files_extent", full.names = T)
  if(length(tmpfile_files_extent) == 1){
    load(tmpfile_files_extent)
  } else {
    files_extent = data.table(file = character(), xmin = numeric(), xmax = numeric(), ymin = numeric(), ymax = numeric())
    tmpfile_files_extent = tempfile(pattern = "tmpfile_files_extent_", fileext = ".RData")
  }
  
  # add files not yet assessed
  files_adjacent_potential = list.files(dirname(path_origin), pattern = type_file, recursive = T, full.names = T) # find all files in other directories next to the current one
  files_adjacent_potential = files_adjacent_potential[!files_adjacent_potential %in% files_extent$file]
  if(length(files_adjacent_potential) > 0){
    files_extent_add = data.table(file = files_adjacent_potential, xmin = as.numeric(NA), xmax = as.numeric(NA), ymin = as.numeric(NA), ymax = as.numeric(NA))
    files_extent = rbind(files_extent, files_extent_add)
  }
  
  # overview over extent of current files, using a specified buffer (can be small in theory, as we are looking for directly adjacent tiles, but should be wide enough to account for pulse density variation)
  outlines_cluster = aggregate(outlines_cluster)
  outlines_cluster = buffer_withsf(outlines_cluster, buffer)
  
  # now compute adjacency
  is.adjacent = integer()
  for(j in 1:nrow(files_extent)){
    if(j %% 100 == 0) cat("Checked",j,"files for adjacency\n")
    
    if(like(files_extent[j]$file,path_origin,fixed = T)){
      is.adjacent_current = F # exclude the currently analyzed files themselves (no need to copy them again)
      is.adjacent = c(is.adjacent, is.adjacent_current)
    } else {
      # check whether we need to update the file information
      if(is.na(files_extent[j]$xmin) | is.na(files_extent[j]$xmax) | is.na(files_extent[j]$ymin) | is.na(files_extent[j]$ymax)){
        header_file_extent_current = readLASheader(files_extent[j]$file);
        files_extent[j]$xmin = header_file_extent_current$`Min X`
        files_extent[j]$xmax = header_file_extent_current$`Max X`
        files_extent[j]$ymin = header_file_extent_current$`Min Y`
        files_extent[j]$ymax = header_file_extent_current$`Max Y`
      }
      
      # check overlap - assumes that crs are the same!!!!
      outlines_file = as.polygons(ext(files_extent[j]$xmin, files_extent[j]$xmax, files_extent[j]$ymin, files_extent[j]$ymax)); terra::crs(outlines_file) = terra::crs(outlines_cluster)
      is.adjacent_current = relate(outlines_cluster, outlines_file,"intersects")
      is.adjacent = c(is.adjacent, is.adjacent_current)
    }
  }
  
  # save temporary file to speed up processing during this session
  save(files_extent, file = tmpfile_files_extent)
  
  files_extent[, adjacent := is.adjacent]
  files_adjacent = files_extent[is.adjacent == 1]$file
  file.copy(files_adjacent, path_moveto, overwrite = T)
  return(files_adjacent)
}

# remove adjacent files again, as long as they don't overlap
remove.adjacent = function(path_tiles, outlines_cluster, type_file, size_tile){
  files_tiles = list.files(path_tiles, pattern = type_file, full.names = T)
  outlines_cluster = aggregate(outlines_cluster)
  outlines_cluster = buffer_withsf(outlines_cluster,-1)
  
  keep.file = lapply(files_tiles, function(x){
    basename_x = gsub(paste0(".",type_file),"",basename(x),fixed = T); xmin = as.numeric(tstrsplit(basename_x,"_")[[1]]);ymin = as.numeric(tstrsplit(basename_x,"_")[[2]]);outlines_x = as.polygons(ext(xmin,xmin+size_tile,ymin,ymin+size_tile)); terra::crs(outlines_x) = terra::crs(outlines_cluster); y = is.related(outlines_cluster, outlines_x,"intersects"); return(y)
  })
  # keep.file = lapply(files_tiles, function(x){
  #   outlines_x = as.polygons(ext(readLAS(x,filter = "-drop_withheld"))); terra::crs(outlines_x) = terra::crs(outlines_cluster); y = is.related(outlines_cluster, outlines_x,"intersects"); return(y)
  # })
  
  keep.file = unlist(keep.file)
  files_removable = files_tiles[!keep.file]
  file.remove(files_removable)
  return(files_removable)
}

first_last_points = function(x, y, step)
{
  nfirst = sum(x == 1)/step
  nlast = sum(x == y)/step
  return(list(nfirst=nfirst, nlast=nlast))
}

# determine pulse density in a m2 grid
get.pulsedensity = function(params_general, path_output = "", type_output = "tif", step, scanangle_abs_max = NULL, keep_first = T, scan_angle_field="ScanAngle"){
  
  # define output directory
  if(path_output == ""){
    path_output = file.path(params_general$path_data,"pulsedensity")
  }
  if(!dir.exists(path_output)) dir.create(path_output)
  
  # pass lastools command to system
  if(!is.voidstring(params_general$path_lastools)){
    return_system = system(
      paste(
        file.path.system(type_os = params_general$type_os, params_general$path_lastools, update.command_lastools("lasgrid", type_architecture = params_general$type_architecture)),
        "-i",
        file.path.system(type_os = params_general$type_os, params_general$path_data,paste0("*.",params_general$type_file)),
        "-cores", params_general$n_cores,
        ifelse(params_general$bbtype == "tile","-use_tile_bb",""),
        ifelse(keep_first == T,"-keep_first","-keep_last"), # better to
        "-step", step,
        ifelse(step > 5, "-point_density_32bit","-point_density_16bit"), # safer version than merely point_density, to avoid overflow for larger cell sizes or extremely high density scans (e.g., drones)
        "-odir", file.path.system(type_os = params_general$type_os, path_output),
        paste0("-o",type_output),
        ifelse(!is.null(scanangle_abs_max), paste0("-keep_scan_angle ",-scanangle_abs_max," ",scanangle_abs_max),""),
        ifelse(params_general$type_os == "Linux","2>&1", "") # crucial for Linux where LAStools seems to write to stderr
      )
    )
    
    if(return_system == 0){
      cat("Successful completion of system command\n")
    } else {
      # logfile_dt = data.table(path = as.character(NA), issue = paste0("WARNING! Pulse density mapping generated non-0 return."))
      # fwrite(logfile_dt, file = params_general$logfile, append = T)
      cat("System command could not be completed\n")
    } 
  } else {
    
    
    cat("Pulse density open source called...\n")
    files.input = list.files(path = params_general$path_data, pattern = "\\.laz", full.names = TRUE)
    
    ctg <- readLAScatalog(files.input)
    opt_output_files(ctg) <- file.path(path_output, "pulsedensity_{ID}")
    
    future::plan(future::multisession, workers = params_general$n_cores)
    
    expr <- bquote(~first_last_points(ReturnNumber, NumberOfReturns, .(step)))
    r <- pixel_metrics(ctg, expr, res = step)
    
    opt_output_files(ctg) <- file.path(path_output, "pulsedensity_scan_angle{ID}")
    
    filter_expr <- as.formula(
      substitute(~field < thresh,
                 list(field = as.name(scan_angle_field), thresh = scanangle_abs_max)
      )
    )
    r <- pixel_metrics(ctg, ~length(Z), res = step, filter = filter_expr)
    
    # expr = bquote(.(as.name(scan_angle_field)) < .(scanangle_abs_max))
    # r <- pixel_metrics(ctg, ~length(Z), res = step, filter=expr)
    
    future::plan(future::sequential)
    
    gc()
    cat("Pulse density open source done!\n")
  }
}

# get number of ground points
get.grounddensity = function(params_general, path_output = "", type_output = "tif", step){
  
  # define output directory
  if(path_output == ""){
    path_output = file.path(params_general$path_data,"grounddensity")
  }
  if(!dir.exists(path_output)) dir.create(path_output)
  
  # pass lastools command to system
  if(!is.voidstring(params_general$path_lastools)){
    return_system = system(
      paste(
        file.path.system(type_os = params_general$type_os, params_general$path_lastools, update.command_lastools("lasgrid", type_architecture = params_general$type_architecture)),
        "-i",
        file.path.system(type_os = params_general$type_os, params_general$path_data,paste0("*.",params_general$type_file)),
        "-cores", params_general$n_cores,
        ifelse(params_general$bbtype == "tile","-use_tile_bb",""),
        "-keep_class 2",
        "-step", step,
        ifelse(step > 5, "-point_density_32bit","-point_density_16bit"), # safer version than merely point_density, to avoid overflow for larger cell sizes or extremely high density scans (e.g., drones)
        "-odir", file.path.system(type_os = params_general$type_os, path_output),
        paste0("-o",type_output),
        ifelse(params_general$type_os == "Linux","2>&1", "") # crucial for Linux where LAStools seems to write to stderr
      )
    )
  } else {
    # 
    # # !!! TOTEST: lidR implemetnation
    # set_lidr_threads(params_general$n_cores)
    # 
    # # read files
    # files.input = list.files(path = params_general$path_data,pattern = paste0("\\.",params_general$type_file), full.names = TRUE)
    # ctg = readLAScatalog(files.input)
    # 
    # # process
    # opt_filter(ctg) = "-drop_withheld -remove_noise -keep_first"
    # opt_chunk_buffer(ctg) = params_general$buffer
    # opt_output_files(ctg) = paste0(path_output,"/{*}")
    # opt_progress(ctg) = FALSE # deactivate rendering of progress
    # 
    # grid_density(ctg, res = step)
  }
}

get.scanangle_abs = function(params_general, path_output = "", type_output = "tif", step, scan_angle_field = "ScanAngle"){
  
  # define output directory
  if(path_output == ""){
    path_output = file.path(params_general$path_data,"scanangle_abs")
  }
  if(!dir.exists(path_output)) dir.create(path_output)
  
  # pass lastools command to system
  if(!is.voidstring(params_general$path_lastools)){
    return_system = system(
      paste(
        file.path.system(type_os = params_general$type_os, params_general$path_lastools, update.command_lastools("lasgrid", type_architecture = params_general$type_architecture)),
        "-i",
        file.path.system(type_os = params_general$type_os, params_general$path_data,paste0("*.",params_general$type_file)),
        "-cores", params_general$n_cores,
        ifelse(params_general$bbtype == "tile","-use_tile_bb",""),
        "-step", step,
        "-scan_angle_abs",
        "-odir", file.path.system(type_os = params_general$type_os, path_output),
        paste0("-o",type_output),
        ifelse(params_general$type_os == "Linux","2>&1", "") # crucial for Linux where LAStools seems to write to stderr
      )
    )
    
    if(return_system == 0){
      cat("Successful completion of system command\n")
    } else {
      # logfile_dt = data.table(path = as.character(NA), issue = paste0("WARNING! Scan angle mapping generated non-0 return."))
      # fwrite(logfile_dt, file = params_general$logfile, append = T)
      cat("System command could not be completed\n")
    } 
  }
  else
  {
    path_data   <- params_general$path_data
    if(!dir.exists(path_output)) dir.create(path_output, recursive = TRUE)
    
    ctg <- readLAScatalog(path_data)
    opt_output_files(ctg) <- file.path(path_output, "scanangle_abs_{ID}")
    
    future::plan(future::multisession, workers = params_general$n_cores)
    r <- pixel_metrics(ctg, bquote(~max(abs(.(as.name(scan_angle_field))))), res = step)
    future::plan(future::sequential)
    
  }
}

# create grids of surface variability
get.surfacevar = function(params_general, path_output = "", name_raster = "surface_var", type_output = "tif", step = 1, metric = "-stddev", arguments_additional = ""){
  # record time
  time_start = Sys.time()
  
  # define output directory
  if(path_output == ""){
    path_output = file.path(params_general$path_data,name_raster)
  }
  if(!dir.exists(path_output)) dir.create(path_output)
  
  if(!is.voidstring(params_general$path_lastools)){
    # create a simple chm, based on highest returns
    return_system = system(
      paste(
        file.path.system(type_os = params_general$type_os, params_general$path_lastools, update.command_lastools("lasgrid", type_architecture = params_general$type_architecture)),
        "-i", file.path.system(type_os = params_general$type_os, params_general$path_data,"*.laz"),
        "-cores", params_general$n_cores,
        "-step", step,
        metric,
        arguments_additional,
        ifelse(params_general$bbtype == "tile","-use_tile_bb",""),
        "-odir", file.path.system(type_os = params_general$type_os, path_output),
        "-otif",
        ifelse(params_general$type_os == "Linux","2>&1", "") # crucial for Linux where LAStools seems to write to stderr
      )
    )
  }
  
  time_end = Sys.time()
  time_processing = difftime(time_end,time_start,units = "mins")
  return(time_processing)
}

get.laserpenetration = function(params_general, path_output = "", type_output = "tif", step, ignore.below = 2){
  
  # define output directory
  if(path_output == ""){
    path_output = file.path(params_general$path_data,"laserpenetration")
  }
  if(!dir.exists(path_output)) dir.create(path_output)
  
  # pass lastools command to system
  if(!is.voidstring(params_general$path_lastools)){
    return_system = system(
      paste(
        file.path.system(type_os = params_general$type_os, params_general$path_lastools, update.command_lastools("lasgrid", type_architecture = params_general$type_architecture)),
        "-i",
        file.path.system(type_os = params_general$type_os, params_general$path_data,paste0("*.",params_general$type_file)),
        "-cores", params_general$n_cores,
        ifelse(params_general$bbtype == "tile","-use_tile_bb",""),
        "-drop_z_below", ignore.below,
        "-keep_first",
        "-step", step,
        "-counter_32bit",
        "-odir", file.path.system(type_os = params_general$type_os, path_output),
        paste0("-o",type_output),
        "-odix", "_first",
        ifelse(params_general$type_os == "Linux","2>&1", "") # crucial for Linux where LAStools seems to write to stderr
      )
    )
    
    return_system = system(
      paste(
        file.path.system(type_os = params_general$type_os, params_general$path_lastools, update.command_lastools("lasgrid", type_architecture = params_general$type_architecture)),
        "-i",
        file.path.system(type_os = params_general$type_os, params_general$path_data,paste0("*.",params_general$type_file)),
        "-cores", params_general$n_cores,
        ifelse(params_general$bbtype == "tile","-use_tile_bb",""),
        "-drop_z_below", ignore.below,
        "-keep_first",
        "-keep_last",
        "-step", step,
        "-counter_32bit",
        "-odir", file.path.system(type_os = params_general$type_os, path_output),
        paste0("-o",type_output),
        "-odix", "_firstlast",
        ifelse(params_general$type_os == "Linux","2>&1", "") # crucial for Linux where LAStools seems to write to stderr
      )
    )
    
    if(return_system == 0){
      cat("Successful completion of system command\n")
    } else {
      # logfile_dt = data.table(path = as.character(NA), issue = paste0("WARNING! Laser penetration mapping generated non-0 return."))
      # fwrite(logfile_dt, file = params_general$logfile, append = T)
      cat("System command could not be completed\n")
    } 
  } else {
    #TODO: lidR implemetnation
    
  }
}

lassplit = function(params_general, path_output = "", update.path = TRUE){
  
  # define output directory
  if(is.voidstring(path_output)){
    path_output = file.path(params_general$path_data,"lassplit")
  }
  if(!dir.exists(path_output)) dir.create(path_output,showWarnings = FALSE)
  
  # pass lastools command to system
  if(!is.voidstring(params_general$path_lastools)){
    return_system = system(
      paste(
        file.path.system(type_os = params_general$type_os, params_general$path_lastools, update.command_lastools("lassplit", type_architecture = params_general$type_architecture)),
        "-i",
        file.path.system(type_os = params_general$type_os, params_general$path_data,paste0("*.",params_general$type_file)),
        "-cores", params_general$n_cores,
        "-merged",
        "-odir", file.path.system(type_os = params_general$type_os, path_output),
        "-o flightline.laz",
        ifelse(params_general$type_os == "Linux","2>&1", "") # crucial for Linux where LAStools seems to write to stderr
      )
    )
    
    if(return_system == 0){
      cat("Successful completion of system command\n")
    } else {
      # logfile_dt = data.table(path = as.character(NA), issue = paste0("WARNING! Lassplit generated non-0 return."))
      # fwrite(logfile_dt, file = params_general$logfile, append = T)
      cat("System command could not be completed\n")
    } 
  }
  
  if(params_general$cleanup == T) cleanup.files(params_general$path_data)
  
  if(update.path == TRUE){
    cat("Amending file path and reindexing\n")
    params_general$path_data = path_output
    params_general$type_file = "laz"
    params_general$bbtype = "tile"
    # lasindex(params_general)
    cat("Amended file path returned\n")
    return(params_general)
  }
}


# retile point cloud for efficient parallel processing
# since v.44: slightly slicker, no dropping of points at that stage anymore, and "extra_pass", in case it can help with memory issues
lastile = function(params_general, path_output = "", size_tile = 500, update.path = TRUE, refine = NULL, size_MB_input = 0, arguments_additional = ""){
  
  cat("lastile\n")
  
  # define output directory
  if(is.voidstring(path_output)){
    path_output = file.path(params_general$path_data,"lastile")
  }
  if(!dir.exists(path_output)) dir.create(path_output,showWarnings = FALSE)
  
  # pass lastools command to system
  if(!is.voidstring(params_general$path_lastools)){
    return_system = system(
      paste(
        file.path.system(type_os = params_general$type_os, params_general$path_lastools, update.command_lastools("lastile", type_architecture = params_general$type_architecture)),
        "-i",
        file.path.system(type_os = params_general$type_os, params_general$path_data,paste0("*.",params_general$type_file)),
        "-cores", params_general$n_cores,
        arguments_additional,
        "-tile_size", size_tile,
        "-buffer", params_general$buffer,
        "-flag_as_withheld",
        ifelse(size_MB_input > 5000,"-extra_pass",""), # added in v.44: attempting to address memory issues (cf. readme)
        ifelse(!is.null(refine), paste0("-refine ",format(refine,scientific = F)),""),
        "-odir", file.path.system(type_os = params_general$type_os, path_output),
        "-olaz",
        ifelse(params_general$type_os == "Linux","2>&1", "") # crucial for Linux where LAStools seems to write to stderr
      )
    )
    
    if(!is.null(refine)){
      params_general_refine = copy(params_general)
      params_general_refine$path_data = path_output
      params_general_refine$type_file = "laz"
      params_general_refine$bbtype = "tile"
      # lasindex(params_general_refine)
      return_system = system(
        paste(
          file.path.system(type_os = params_general$type_os, params_general$path_lastools, update.command_lastools("lastile", type_architecture = params_general$type_architecture)),
          "-i",
          file.path.system(type_os = params_general$type_os, path_output,paste0("*_",size_tile,".laz")),
          "-cores", params_general$n_cores,
          "-refine_tiles ",format(refine,scientific = F),
          "-odir", file.path.system(type_os = params_general$type_os, path_output),
          "-olaz",
          ifelse(params_general$type_os == "Linux","2>&1", "") # crucial for Linux where LAStools seems to write to stderr
        )
      )
    }
    
    
    if(return_system == 0){
      cat("Successful completion of system command\n")
    } else {
      # logfile_dt = data.table(path = as.character(NA), issue = paste0("WARNING! Lastile generated non-0 return."))
      # fwrite(logfile_dt, file = params_general$logfile, append = T)
      cat("System command could not be completed\n")
    } 
  } else {
    
    # lidR implementation
    # this might not be necessary in practice, as lidR does this under the hood, to be checked
    
    # read files
    files.input = list.files(path = params_general$path_data,pattern = paste0("\\.",params_general$type_file), full.names = TRUE)
    ctg = readLAScatalog(files.input)
    
    # process data
    # opt_filter(ctg) = paste0("-drop_withheld", paste0(" -drop_class 7 18 ", paste(class_rm, collapse = " ")), ifelse(length(exclass_rm) > 0, paste0(" -drop_extended_class ", paste(exclass_rm, collapse = " ")),"")) # new in v.47: remove extra noise classes or otherwise from scan (necessary for IGN France, for example, where artefacts/noise is usually marked with 65, or even 28) # removed due to inconsistency; this is the wrong location for this step
    opt_output_files(ctg) = paste0(path_output, "/{XLEFT}_{YBOTTOM}")
    opt_progress(ctg) = FALSE # deactivate rendering of progress
    opt_laz_compression(ctg) = TRUE
    opt_chunk_size(ctg) = size_tile
    opt_chunk_buffer(ctg) = params_general$buffer #TODO to be checked
    catalog_retile(ctg)
  }
  
  if(params_general$cleanup == T) cleanup.files(params_general$path_data)
  
  if(update.path == TRUE){
    cat("Amending file path and reindexing\n")
    params_general$path_data = path_output
    params_general$type_file = "laz"
    params_general$bbtype = "tile"
    # lasindex(params_general)
    cat("Amended file path returned\n")
    return(params_general)
  }
}


# general information function
# the idea is to first compute summary statistics per las file (lasinfo), then summarize them further across the whole acquisition
# !!! TODO: return median of pulse density, not just mean
# !!! TODO: this could vastly improve the application of the spikefree algorithm, as the mean may overestimate pulse density under highly irregular sampling (e.g. drone scans with high densities at turning points)
lasinfo = function(params_general, path_output = ""){
  
  # lasinfo files are only created with lastools, the summary, however, is created for both files
  summary_bytile = NULL
  
  # define output directory
  if(path_output == ""){
    path_output = file.path(params_general$path_data,"lasinfo")
  }
  if(!dir.exists(path_output)) dir.create(path_output)
  
  if(!is.voidstring(params_general$path_lastools)){
    
    # pass lastools command to system
    return_system = system(
      paste(
        file.path.system(type_os = params_general$type_os, params_general$path_lastools, update.command_lastools("lasinfo", type_architecture = params_general$type_architecture)),
        "-i",
        file.path.system(type_os = params_general$type_os, params_general$path_data,paste0("*.",params_general$type_file)),
        "-cores", params_general$n_cores,
        "-no_vlrs", # remove user specific extra information
        "-compute_density",
        "-drop_withheld",
        "-odir", file.path.system(type_os = params_general$type_os, path_output),
        "-otxt",
        ifelse(params_general$type_os == "Linux","2>&1", "") # crucial for Linux where LAStools seems to write to stderr
      )
    )
    
    if(return_system == 0){
      cat("Successful completion of system command\n")
    } else {
      # logfile_dt = data.table(path = as.character(NA), issue = paste0("WARNING! Lasinfo generated non-0 return."))
      # fwrite(logfile_dt, file = params_general$logfile, append = T)
      cat("System command could not be completed\n")
    } 
    
    if(return_system == 0){
      files_info = list.files(path_output, pattern = ".txt",full.names = TRUE)
      summary_bytile = rbindlist(lapply(files_info, summarize.lasinfo))
      summary_bytile = summary_bytile[area > 0] # new in v.47
    }
  } else {
    
    cat("open source info called...\n")
    files.input = list.files(path = params_general$path_data, pattern = paste0("\\.",params_general$type_file), full.names = TRUE)
    
    ctg <- readLAScatalog(files.input)
    
    opt_chunk_size(ctg) <- 0  # file-by-file
    opt_chunk_buffer(ctg) <- 0
    opt_progress(ctg) <- TRUE
    opt_independent_files(ctg) <- TRUE
    
    future::plan(future::multisession, workers = params_general$n_cores)
    
    res_list <- catalog_apply(ctg, ctg_summarize_chunk)
    summary_bytile <- rbindlist(res_list, use.names = TRUE, fill = TRUE)
    
    future::plan(future::sequential)
    
    cat("open source info done!\n")
  }
  
  # get current date time to determine maximum time stamp possible
  GPS_OFFSET = 1e9
  date_origin = as.Date("1980-01-06")
  gps_max = as.Date(Sys.time())
  gps_max = as.numeric(difftime(gps_max, date_origin, units = "s")) - GPS_OFFSET
  
  # whole-acquisition summary statistics
  summary_full = NULL
  if(!is.null(summary_bytile)){
    cat("Computing summary statistics")
    
    summary_full = data.table(
      n_files = nrow(summary_bytile),
      classified = ifelse(all(summary_bytile[!is.na(classified)]$classified),TRUE,FALSE),
      encoding_global = max(summary_bytile$encoding_global),
      mingps = ifelse(nrow(summary_bytile[mingps > 0 & mingps < gps_max]) > 0, min(summary_bytile[mingps > 0 & mingps < gps_max]$mingps), min(summary_bytile$mingps)),
      maxgps = ifelse(nrow(summary_bytile[maxgps > 0 & maxgps < gps_max]) > 0, max(summary_bytile[maxgps > 0 & maxgps < gps_max]$maxgps), max(summary_bytile$maxgps)),
      points_mean = weighted.mean(summary_bytile$points, w = summary_bytile$area, na.rm = T),
      points_ground_mean = weighted.mean(summary_bytile$points_ground, w = summary_bytile$area, na.rm = T),
      fraction_ground_mean = weighted.mean(summary_bytile$fraction_ground, w = summary_bytile$area, na.rm = T),
      density_points_mean = weighted.mean(summary_bytile$density_points, w = summary_bytile$area, na.rm = T),
      density_pulses_mean = weighted.mean(summary_bytile$density_pulses, w = summary_bytile$area, na.rm = T),
      spacing_pulses_mean = weighted.mean(summary_bytile$spacing_pulses, w = summary_bytile$area, na.rm = T),
      points_sd = sqrt(wtd.var(summary_bytile$points, weights = summary_bytile$area, na.rm = T)),
      points_ground_sd = sqrt(wtd.var(summary_bytile$points_ground, weights = summary_bytile$area, na.rm = T)),
      fraction_ground_sd = sqrt(wtd.var(summary_bytile$fraction_ground, weights = summary_bytile$area, na.rm = T)),
      density_points_sd = sqrt(wtd.var(summary_bytile$density_points, weights = summary_bytile$area, na.rm = T)),
      density_pulses_sd = sqrt(wtd.var(summary_bytile$density_pulses, weights = summary_bytile$area, na.rm = T)),
      spacing_pulses_sd = sqrt(wtd.var(summary_bytile$spacing_pulses, w = summary_bytile$area, na.rm = T))
    )
    summary_full[
      ,`:=`(
        points_cv = points_sd/points_mean,
        points_ground_cv = points_ground_sd/points_ground_mean,
        fraction_ground_cv = fraction_ground_sd/fraction_ground_mean,
        density_points_cv = density_points_sd/density_points_mean,
        density_pulses_cv = density_pulses_sd/density_pulses_mean,
        spacing_pulses_cv = spacing_pulses_sd/spacing_pulses_mean
      )
    ]
    save(summary_bytile, file = file.path(params_general$path_data,"summary_bytile.RData"))
    save(summary_full, file = file.path(params_general$path_data,"summary_full.RData"))
  }
  return(summary_full)
}

# two helper functions to further summarize initial information (LAStools and lidR version)
# lasinfo does not seem to drop the withheld points correctly for the summary statistics, so statistics are not strictly comparable due to buffers
# !!! TODO: merge into one function
summarize.lasinfo = function(path){
  
  name_file = basename(path)
  file_info = readLines(path)
  
  if(length(file_info) > 0){
    points = ifelse(as.numeric(strsplit(trimws(file_info[grep("number of point records", file_info,fixed = TRUE)])," ")[[1]][8]) == 0,tryCatch(as.numeric(strsplit(trimws(file_info[grep("number of point records", file_info,fixed = TRUE)])," ")[[2]][6]), error=function(err) 0),as.numeric(strsplit(trimws(file_info[grep("number of point records", file_info,fixed = TRUE)])," ")[[1]][8])) # addition in v.46 to accommodate extended records in OntarioSPL files
    points_ground = tryCatch(as.numeric(strsplit(trimws(file_info[grep("ground (2)", file_info,fixed = TRUE)])," ")[[1]][1]), error=function(err) 0)
    fraction_ground = points_ground/points
    density_points = tryCatch(as.numeric(strsplit(trimws(file_info[grep("point density", file_info,fixed = TRUE)])," ")[[1]][5]), error=function(err) as.numeric(NA))
    density_pulses = tryCatch(as.numeric(strsplit(trimws(file_info[grep("point density", file_info,fixed = TRUE)])," ")[[1]][8]), error=function(err) as.numeric(NA))
    spacing_pulses = tryCatch(sqrt(1/density_pulses), error=function(err) as.numeric(NA))
    area = tryCatch(as.numeric(strsplit(strsplit(trimws(file_info[grep("covered area in square", file_info,fixed = TRUE)])," ")[[1]][6],"/")[[1]][1]), error=function(err) as.numeric(NA))
    encoding_global = tryCatch(as.integer(strsplit(gsub("\\s+", " ",trimws(file_info[grep("global_encoding", file_info,fixed = TRUE)]))," ")[[1]][2]), error=function(err) as.integer(NA))
    mingps = tryCatch(as.numeric(strsplit(trimws(file_info[grep("gps_time", file_info,fixed = TRUE)])," ")[[1]][2]), error=function(err) as.numeric(NA))
    maxgps = tryCatch(as.numeric(strsplit(trimws(file_info[grep("gps_time", file_info,fixed = TRUE)])," ")[[1]][3]), error=function(err) as.numeric(NA))
    
    return(data.table(file = name_file, classified = ifelse(!is.na(points_ground) & !is.null(points_ground) & points_ground != "", TRUE, FALSE), points = points, points_ground = points_ground, fraction_ground = fraction_ground, density_points = density_points, density_pulses = density_pulses, spacing_pulses = spacing_pulses, area = area, encoding_global = encoding_global, mingps = mingps, maxgps = maxgps))
  } else {
    return(data.table(file = name_file, classified = as.logical(NA), points = as.integer(NA), points_ground = as.integer(NA), fraction_ground = as.numeric(NA), density_points = as.numeric(NA), density_pulses = as.numeric(NA), spacing_pulses = as.numeric(NA), area = as.numeric(NA, encoding_global = as.integer(NA), mingps = as.numeric(NA), maxgps = as.numeric(NA))))
  }
}

# deduplicate data
# strict scheme is applied, i.e. only exact duplicates (x, y, z) are removed, and not just x y duplicates (default in LAStools)
lasduplicate = function(params_general, path_output = "",  update.path = TRUE){
  
  # browser()
  # browser()
  #   
  # define output directory
  if(path_output == ""){
    path_output = file.path(params_general$path_data,"lasduplicate")
  }
  if(!dir.exists(path_output)) dir.create(path_output)
  
  # pass lastools command to system
  if(!is.voidstring(params_general$path_lastools)){
    return_system = system(
      paste(
        file.path.system(type_os = params_general$type_os, params_general$path_lastools, update.command_lastools("lasduplicate", type_architecture = params_general$type_architecture)),
        "-i",
        file.path.system(type_os = params_general$type_os, params_general$path_data, paste0("*.",params_general$type_file)),
        "-cores", params_general$n_cores,
        "-unique_xyz",
        "-odir", file.path.system(type_os = params_general$type_os, path_output),
        "-olaz",
        ifelse(params_general$type_os == "Linux","2>&1", "") # crucial for Linux where LAStools seems to write to stderr
      )
    )
    
    if(return_system == 0){
      cat("Successful completion of system command\n")
    } else {
      # logfile_dt = data.table(path = as.character(NA), issue = paste0("WARNING! Lasduplicate generated non-0 return."))
      # fwrite(logfile_dt, file = params_general$logfile, append = T)
      cat("System command could not be completed\n")
    } 
    
    # lasindex(params_general)
    
  } else {
    
    cat("lasduplicate open source called...\n")
    
    output_filename <- file.path(path_output, "*.laz")
    
    pipeline <- lasR::reader(filter = lasR::drop_duplicates()) + lasR::write_las(output_filename)
    
    path_r <- file.path(params_general$path_data)
    files <- list.files(
      path_r, 
      pattern = "\\.laz$", 
      ignore.case = TRUE, 
      full.names = TRUE
    )
    if (params_general$type_os == "Windows") {
      files_lastools <- paste0('"', files, '"')
    } else {
      files_lastools <- files
    }
    # files <- list.files(file.path.system(type_os = params_general$type_os, params_general$path_data), 
    #                     pattern = "*.laz$", full.names = TRUE)
    
    lasR::exec(
      pipeline,
      on = files,
      ncores = params_general$n_cores,
      with = list(chunk = size_tile),
      progress = T
    )
    
    # # --- 4. Run parallel loop ---
    # foreach(
    #   filename = files,
    #   .combine = c
    #   # .packages = c("lasR", "tools"),
    #   # .export = c("lasduplicate_custom_function")
    # ) %dopar% {
    #   lasduplicate_custom_function(
    #     file = filename,
    #     path_output = path_output,
    #     size_tile = size_tile)
    # }
    
    # stopCluster(cl)
    cat("lasduplicate open source done!\n")
  }
  
  if(params_general$cleanup == T) cleanup.files(params_general$path_data)
  
  if(update.path == TRUE){
    cat("Amending file path and reindexing\n")
    params_general$path_data = path_output
    params_general$type_file = "laz"
    cat("Amended file path returned\n")
    return(params_general)
  }
}

# denoise
# parameters (step/isolated) should always be chosen based on pulse density and what kind of noise one would expect
# a common example for an upper limit on noise would likely be birds
# e.g. assume 5 pulses per m2 and a large bird (1 m2 of exposed wing area), then we would expect the bird to generate 5 returns (isolated = 5); a buffer of 3m (step = 3) between the bird and canopy would probably make sense
# NOTE: rain/clouds are difficult to denoise, as they could generate many false returns, we likely have to rely on the data providers to have done a good pre-cleaning, or need to do manual assessments in this case

lasnoise = function(params_general, path_output = "", step = 3, isolated = 3, update.path = TRUE){
  
  # define output directory
  if(path_output == ""){
    path_output = file.path(params_general$path_data,"lasnoise")
  }
  if(!dir.exists(path_output)) dir.create(path_output)
  
  # pass lastools command to system
  if(!is.voidstring(params_general$path_lastools)){
    return_system = system(
      paste(
        file.path.system(type_os = params_general$type_os, params_general$path_lastools, update.command_lastools("lasnoise", type_architecture = params_general$type_architecture)),
        "-i",
        file.path.system(type_os = params_general$type_os, params_general$path_data,paste0("*.",params_general$type_file)),
        "-cores", params_general$n_cores,
        "-step", step,
        "-isolated", isolated,
        "-remove_noise",
        "-odir", file.path.system(type_os = params_general$type_os, path_output),
        "-olaz",
        ifelse(params_general$type_os == "Linux","2>&1", "") # crucial for Linux where LAStools seems to write to stderr
      )
    )
    
    if(return_system == 0){
      cat("Successful completion of system command\n")
    } else {
      # logfile_dt = data.table(path = as.character(NA), issue = paste0("WARNING! Lasnoise generated non-0 return."))
      # fwrite(logfile_dt, file = params_general$logfile, append = T)
      cat("System command could not be completed\n")
    } 
    
  } else {
    
    files.input = list.files(path = params_general$path_data,pattern = paste0("\\.",params_general$type_file), full.names = TRUE)
    ctg = readLAScatalog(files.input)
    
    opt_filter(ctg) = "-drop_withheld"
    opt_chunk_buffer(ctg) = params_general$buffer
    opt_output_files(ctg) = paste0(path_output,"/{*}")
    opt_progress(ctg) = FALSE
    opt_laz_compression(ctg) = TRUE
    classify_noise(ctg, ivf(res = step, n = isolated))
  }
  
  if(params_general$cleanup == T) cleanup.files(params_general$path_data)
  
  if(update.path == TRUE){
    cat("Amending file path and reindexing\n")
    params_general$path_data = path_output
    params_general$type_file = "laz"
    # lasindex(params_general)
    cat("Amended file path returned\n")
    return(params_general)
  }
}

# create DTM/DSM
# step size is the initial [!] step size for the algorithm to find ground points (i.e. find ground points within a 10 by 10 m window for step = 10); 
# sub is subwindows for refinement of ground characterization; 
# if both are set to NULL, the default parameters are used
lasground_new = function(params_general, path_output = "", step = NULL, sub = NULL, divisor_bulge = 5.0, update.path = TRUE, arguments_additional = ""){
  
  # define output directory
  if(path_output == ""){
    path_output = file.path(params_general$path_data,"lasground_new")
  }
  if(!dir.exists(path_output)) dir.create(path_output)
  
  # pass lastools command to system
  if(!is.voidstring(params_general$path_lastools)){
    return_system = system(
      paste(
        file.path.system(type_os = params_general$type_os, params_general$path_lastools, update.command_lastools("lasground_new", type_architecture = params_general$type_architecture)),
        "-i",
        file.path.system(type_os = params_general$type_os, params_general$path_data,paste0("*.",params_general$type_file)),
        "-cores", params_general$n_cores,
        ifelse(!is.null(step),paste0("-step ",round(step, 3)),""),
        ifelse(!is.null(sub),paste0("-sub ",round(sub, 3)),""),
        ifelse(!is.null(step),paste0("-bulge ",round(step/divisor_bulge, 3)),""),
        arguments_additional,
        "-odir", file.path.system(type_os = params_general$type_os, path_output),
        "-olaz",
        ifelse(params_general$type_os == "Linux","2>&1", "") # crucial for Linux where LAStools seems to write to stderr
      )
    )
    
    if(return_system == 0){
      cat("Successful completion of system command\n")
    } else {
      # logfile_dt = data.table(path = as.character(NA), issue = paste0("WARNING! Lasground_new generated non-0 return."))
      # fwrite(logfile_dt, file = params_general$logfile, append = T)
      cat("System command could not be completed\n")
    } 
  } else {
    
    files <- list.files(path = params_general$path_data, pattern = paste0("\\.", params_general$type_file, "$"), full.names = TRUE)
    
    output_filename <- paste0(path_output, "/*_lasgroundnew.laz")
    
    pipeline = lasR::classify_with_ptd() + lasR::write_las(output_filename)
    
    lasR::exec(
      pipeline,
      on = files,
      ncores = params_general$n_cores,
      progress = T
    )
    
  }
  
  if(params_general$cleanup == T) cleanup.files(params_general$path_data)
  
  if(update.path == TRUE){
    cat("Amending file path and reindexing\n")
    params_general$path_data = path_output
    params_general$type_file = "laz"
    # lasindex(params_general)
    cat("Amended file path returned\n")
    return(params_general)
  }
}

get.middle = function(x){x[ceiling(length(x)/2)]}

get.diff_aspect = function(x){
  x = x[!is.na(x)]
  x_middle = get.middle(x)
  x_shifted1 = x - 360 
  x_shifted2 = x + 360
  x_diffmiddle = abs(x - x_middle)
  x_diffmiddle1 = abs(x_shifted1 - x_middle)
  x_diffmiddle2 = abs(x_shifted2 - x_middle)
  xdiff = pmin(x_diffmiddle, x_diffmiddle1, x_diffmiddle2)
  xdiff = mean(xdiff)
  return(xdiff)
}

get.areas_steep = function(params_general, path_output = "", step, slope_cutoffs = c(45,60), crs_scan){
  
  # define output directory
  if(path_output == ""){
    path_output = file.path(params_general$path_data, "areas_steep")
  }
  if(!dir.exists(path_output)) dir.create(path_output)
  
  # create a basic dtm
  las2dem(params_general = params_general, path_output = file.path(path_output, "dtm_lasdef"), name_raster = "dtm_lasdef", kill = 200, step = step, type_output = "tif", option = "dtm")
  path_dtm_lasdef = file.path(path_output, "dtm_lasdef")
  files_dtm_lasdef = list.files.nonzero(path = path_dtm_lasdef, pattern = ".tif", full.names = TRUE)
  dtm_lasdef = vrt(files_dtm_lasdef); terra::crs(dtm_lasdef) = crs_scan
  writeRaster(dtm_lasdef, filename = file.path(path_output,paste0("dtm_lasdef.tif")), overwrite = T) # written inside the processing pipeline folder structure
  if (params_general$cleanup == T) file.remove(files_dtm_lasdef)
  
  # create a basic dtm from highest ground points
  make.dtm_highest(params_general = params_general, path_output = file.path(path_output, "dtm_highest_basic"), name_raster = "dtm_highest_basic", type_output = "tif", step = step, subcircle = 0.1)
  path_dtm_highest_basic = file.path(path_output, "dtm_highest_basic")
  files_dtm_highest_basic = list.files.nonzero(path = path_dtm_highest_basic, pattern = ".tif", full.names = TRUE)
  dtm_highest_basic = vrt(files_dtm_highest_basic); terra::crs(dtm_highest_basic) = crs_scan
  writeRaster(dtm_highest_basic, filename = file.path(path_output,paste0("dtm_highest_basic.tif")), overwrite = T) # written inside the processing pipeline folder structure
  if (params_general$cleanup == T) file.remove(files_dtm_highest_basic)
  
  # create a basic dsm from highest points
  make.dsm_highest(params_general = params_general, path_output = file.path(path_output, "dsm_highest_basic"), name_raster = "dsm_highest_basic", type_output = "tif", step = step, subcircle = 0.1)
  path_dsm_highest_basic = file.path(path_output, "dsm_highest_basic")
  files_dsm_highest_basic = list.files.nonzero(path = path_dsm_highest_basic, pattern = ".tif", full.names = TRUE)
  dsm_highest_basic = vrt(files_dsm_highest_basic); terra::crs(dsm_highest_basic) = crs_scan
  writeRaster(dsm_highest_basic, filename = file.path(path_output,paste0("dsm_highest_basic.tif")), overwrite = T) # written inside the processing pipeline folder structure
  if (params_general$cleanup == T) file.remove(files_dsm_highest_basic)
  
  #%%%%%%%%%%%%%%%%#
  # Basic DTM mask #
  #%%%%%%%%%%%%%%%%#
  # we create a slope mask to refine classification on steep slopes
  # all areas classified in this way will be classified by rotating the point cloud
  dtm_agg = aggregate(dtm_lasdef, fact = 5, fun = "mean", na.rm = T) # for speed aggregate first with a factor of 5
  mat_circular = focalMat(dtm_agg, 15, type = "circle") # draw a 25m circle around each point
  mat_circular[mat_circular != 0] = 1
  dtm_agg = focal(dtm_agg, w = mat_circular, fun = "mean", na.rm = T)
  
  # # alternative option: Gaussian filter
  # dtm_agg_nona = ifel(is.na(dtm_agg),NA,1)
  # mat_Gauss = focalMat(dtm_agg, 25, type = "Gauss")
  # dtm_agg = focal(dtm_agg, w = mat_Gauss, fun = "sum", na.rm = T)
  # dtm_agg_nona = focal(dtm_agg_nona, w = mat_Gauss, fun = "sum", na.rm = T)
  # dtm_agg = dtm_agg/dtm_agg_nona
  
  slope_dtm = terrain(dtm_agg, v = "slope", neighbors = 8)
  areas_steep_dtm = ifel(slope_dtm >= slope_cutoffs[1],1,0)
  areas_steep_dtm = focal(areas_steep_dtm, w = 3, fun = "max", na.rm = T)
  
  areas_steeper_dtm = ifel(slope_dtm >= slope_cutoffs[2],1,0)
  # areas_steeper_dtm = focal(areas_steeper_dtm, w = 3, fun = "max", na.rm = T)
  
  #%%%%%%%%%%%%%%%%%%%#
  # Advanced DTM mask #
  #%%%%%%%%%%%%%%%%%%%#
  # there is still one type of area that we would not find like this: areas where the initial ground classification has cut off a terrain bulge
  # these areas typically have no ground points whatsoever and steep slopes in the DSM (which differentiates them from badly sampled areas in otherwise flat forests)
  dsm_agg = aggregate(dsm_highest_basic, fact = 5, fun = "mean", na.rm = T) # for speed aggregate first with a factor of 5
  
  # simple mean filter
  mat_circular = focalMat(dsm_agg, 25, type = "circle") # draw a 25m circle around each point
  mat_circular[mat_circular != 0] = 1
  dsm_agg = focal(dsm_agg, w = mat_circular, fun = "mean", na.rm = T)
  
  # # alternative: Gaussian filter
  # dsm_agg_nona = ifel(is.na(dsm_agg),NA,1)
  # mat_Gauss = focalMat(dsm_agg, 25, type = "Gauss")
  # dsm_agg = focal(dsm_agg, w = mat_Gauss, fun = "sum", na.rm = T)
  # dsm_agg_nona = focal(dsm_agg_nona, w = mat_Gauss, fun = "sum", na.rm = T)
  # dsm_agg = dsm_agg/dsm_agg_nona
  
  slope_dsm = terrain(dsm_agg, v = "slope", neighbors = 8)
  areas_steep_dsm = ifel(slope_dsm >= slope_cutoffs[1],1,0)
  areas_steep_dsm = focal(areas_steep_dsm, w = 3, fun = "max", na.rm = T)
  
  areas_steeper_dsm = ifel(slope_dsm >= slope_cutoffs[2],1,0)
  
  # # use the DTM_highest to define areas without ground points (NA)
  # mat_circular = focalMat(dtm_highest_basic, 10, type = "circle")
  # mat_circular[mat_circular != 0] = 1
  # # mat_circular[mat_circular == 0] = NA
  # ground_nalarge = focal(dtm_highest_basic, w = mat_circular, fun = "mean", na.rm = T)
  # 
  # # build up edges
  # mat_circular = focalMat(dtm_highest_basic, 5, type = "circle")
  # mat_circular[mat_circular != 0] = 1
  # ground_nalarge = focal(ground_nalarge, w = mat_circular, fun = "mean", na.rm = F)
  # ground_nalarge = focal(ground_nalarge, w = mat_circular, fun = "mean", na.rm = F)
  # ground_nalarge = ifel(is.na(ground_nalarge),1,0)
  # ground_nalarge = aggregate(ground_nalarge, fact = 5, fun = "mean",na.rm = T)
  # 
  # # find problematic areas
  # areas_misclassified = ifel(areas_steep_dsm == 1 & ground_nalarge, 1, 0)
  # polys_misclassified = as.polygons(areas_misclassified, values = T)
  # names(polys_misclassified) = "misclassified"
  # polys_misclassified = disagg(polys_misclassified[polys_misclassified$misclassified == 1])
  # polys_misclassified$size = expanse(polys_misclassified)
  # polys_misclassified =  polys_misclassified[polys_misclassified$size >= 1000]
  # 
  # if(nrow(polys_misclassified) == 0){
  #   # remove misclassification entirely
  #   areas_misclassified = clamp(areas_misclassified, upper = 0, values = T)
  # } else {
  #   # mask out small areas
  #   areas_misclassified = terra::mask(areas_misclassified, polys_misclassified, updatevalue = 0)
  # 
  #   # and build up the edges once more
  #   areas_misclassified = focal(areas_misclassified, w = 3, fun = "max", na.rm = T)
  # }
  # 
  # areas_steep = ifel(areas_misclassified > 0 | areas_steeper_dtm == 1, 2, areas_steep_dtm)
  
  areas_steep = ifel(areas_steep_dtm == 1 | areas_steep_dsm == 1, 1, 0)
  areas_steeper = ifel(areas_steeper_dtm == 1 | areas_steeper_dsm == 1, 1, 0)
  areas_steep = ifel(areas_steeper == 1, 2, areas_steep)
  areas_steep = ifel(areas_steep == 0, NA, areas_steep)
  
  # check whether there area any steep areas before returning shapefile
  has_areas_steep = as.numeric(global(areas_steep, "max", na.rm = T))
  
  if(has_areas_steep > 0){
    file_areas_steep = file.path(path_output, "areas_steep.tif")
    writeRaster(areas_steep, filename = file_areas_steep, overwrite = T)
    return(file_areas_steep)
  } else {
    return(NULL)
  }
}

# new in v.48: refine ground classification to better represent edges/ridges/steep slopes/cliffs
# note that there is likely a robustness/accuracy tradeoff; to improve slope representation, finer sampling is required, but this may not be as robust to pulse density as the default LAStools DTM
refine.ground = function(params_general, path_output = "", update.path = TRUE){
  # define output directory
  if(path_output == ""){
    path_output = file.path(params_general$path_data, "refine.cg")
  }
  if(!dir.exists(path_output)) dir.create(path_output)
  
  # process in parallel
  nbcluster = params_general$n_cores
  cl = makeCluster(nbcluster)
  clusterExport(cl, varlist = list("get.middle","get.diff_aspect","refine.ground_tile","file.path.system","update.command_lastools")) # export function, v.42 added buffer_withsf
  clusterEvalQ(
    cl,
    {
      library(data.table) # for %like% operator etc.
      library(parallel)
      library(terra)
      library(lidR)
    }
  )
  on.exit(stopCluster(cl))  # on.exit is also triggered when an error occurs
  
  # refine the ground point classification in steep areas
  # simple procedure:
  # to avoid single large tiles blocking processing towards the end, we sort by tile size
  tiles_pointcloud = data.table(file_tile = list.files(params_general$path_data, pattern = paste0(".",params_general$type_file), full.names = T))
  tiles_pointcloud[, size_file := file.size(file_tile)]
  setorder(tiles_pointcloud, -size_file) # sort in descending order so that smallest files are dealt with last
  tiles_pointcloud = tiles_pointcloud$file_tile
  
  # there could also be a function to determine which tiles actually overlap with steep areas, but for now we only test this within the function "refine.ground_tile" itself.
  # this may lead to suboptimal parallel computing performance in some cases where the initial allocation of tiles to cpus is unbalanced (e.g. most difficult tiles initially allocated to a single cpu), but it's cleaner from  programming perspective, as we need to determine tile extent for subtiles in refine.ground_tile anyways
  
  # alternative for debugging purposes using a loop; commented and kept in the code in case needed
  # j = 58
  # for(j in 1:length(tiles_pointcloud)){
  #   tile_pointcloud = tiles_pointcloud[j]
  #   cat(j,"Tile:",tile_pointcloud,"\n")
  #   refine.ground_tile(tile_pointcloud = tile_pointcloud, params_general = params_general, path_output = path_output)
  # }
  
  parLapplyLB(cl, tiles_pointcloud, refine.ground_tile, params_general = params_general, path_output = path_output)
  
  if(params_general$cleanup == T) cleanup.files(params_general$path_data)
  
  if(update.path == TRUE){
    cat("Amending file path and reindexing\n")
    params_general$path_data = path_output
    params_general$type_file = "laz"
    #lasindex(params_general)
    cat("Amended file path returned\n")
    return(params_general)
  }
}

# the basic principles are as follows:
# 
# small memory requirements, as we always directly add reclassified points to the point cloud
# this is written to be robust to pulse density variation, so will not produce the best possible DTM, but rather the most robust, best possible DTM; for very high pulse densities and good ground sampling, other DTM algorithms (-nature or -wilderness) might be useful
# radius_dsm needs to be set quite high/conservatively so that gradients across large crowns are not mistakenly identified as slopes on a mountain; 25 m is probably safe; the only problem could occur with very very large tree crowns on steep slope, a gap at the bottom of the slope (so that the crown gradient and the dtm gradient overlap) and missing ground returns on the slope; but if that happens, then clearly the lidar scan is not good enough, because you need at least some ground returns on steep slopes
# radius_dtm can be set less conservatively, as this likely indicates actual topographic variation. Still, we don't want tiny dips or spikes in the terrain to lead to reclassification, so we set it to 15

# this is the ground refinement function
# it takes an already classified point cloud (tile_pointcloud) and refines the classification in steep areas
# we employ a two-step refinement
#   1/ In areas of 45 degree or more, we refine the classification of overhanging areas by slightly rotating the point cloud in 8 cardinal directions
#   2/ In areas of 60 degree or more or in areas of 45 degrees with ridges, we employ rotate point cloud in 8 cardinal directions, but also employ lastools -nature option 
# all additional ground points are saved as classs 8

# params_general = data.table(
#   path_lastools = "PATH/TO/LASTOOLS",
#   path_data = "PATH/TO/DATA",
#   type_file = "laz",
#   bbtype = "",
#   type_os = "Windows",
#   type_architecture = "32",
#   n_cores = 6,
#   buffer = 50,
#   cleanup = F,
#   use.blast2dem = T
# )

refine.ground_tile = function(tile_pointcloud, params_general, path_output, nbincrements_xy = 8, radius_dtm = 25){
  
  # we create a folder for the tile
  name_tile = gsub(paste0(".",params_general$type_file),"", basename(tile_pointcloud), fixed = T)
  dir_tile = file.path(path_output, name_tile)
  if(!dir.exists(dir_tile)) dir.create(dir_tile)
  
  # we extract reference ground points
  # over the course of this function, we will gradually add further files with ground points (all ending on _groundextra.laz)
  # these extra ground points will be classified with class 8 
  tile_ground = file.path(dir_tile, paste0(name_tile,"_ground.laz"))
  
  return_system = system(
    paste(
      file.path.system(type_os = params_general$type_os, params_general$path_lastools, update.command_lastools("las2las", type_architecture = params_general$type_architecture)),
      "-i", file.path.system(type_os = params_general$type_os, tile_pointcloud),
      "-cores 1",
      "-keep_class 2",
      "-o", file.path.system(type_os = params_general$type_os, tile_ground),
      ifelse(params_general$type_os == "Linux","2>&1", "") # crucial for Linux where LAStools seems to write to stderr
    )
  )
  
  if(!file.exists(tile_ground)){
    # very rarely it happens that a tile only contains edge points that are not classified as ground, in that case we simply copy
    file.copy(tile_pointcloud, file.path(path_output, paste0(name_tile, ".laz")))
  } else {
    # we then compute the slope of the ground in two ways
    # 1/ via a rudimentary DTM, just from the highest ground points
    return_system = system(
      paste(
        file.path.system(type_os = params_general$type_os, params_general$path_lastools, update.command_lastools("lasgrid", type_architecture = params_general$type_architecture)),
        "-i", file.path.system(type_os = params_general$type_os, tile_pointcloud),
        "-cores 1",
        "-step", 1, # 1 m step size (fixed)
        "-highest",
        "-keep_class 2 8",
        "-subcircle", 1.05,
        "-o", file.path.system(type_os = params_general$type_os, dir_tile, paste0(name_tile,"_dtm_highest.tif")),
        ifelse(params_general$type_os == "Linux","2>&1", "") # crucial for Linux where LAStools seems to write to stderr
      )
    )
    file_dtm_highest = file.path(dir_tile, paste0(name_tile,"_dtm_highest.tif"))
    
    # 2/ via a rudimentary TIN-DTM
    lasgorithm_dem = ifelse(params_general$use.blast2dem == T, "blast2dem", "las2dem")
    
    return_system = system(
      paste(
        file.path.system(type_os = params_general$type_os, params_general$path_lastools, update.command_lastools(lasgorithm_dem, type_architecture = params_general$type_architecture)),
        "-i", file.path.system(type_os = params_general$type_os, tile_pointcloud),
        "-cores 1",
        # "-kill", 100, # maximum size of triangle
        "-step", 1, # 1 m step size
        "-keep_class 2 8",
        "-o", file.path.system(type_os = params_general$type_os, dir_tile, paste0(name_tile,"_dtm.tif")),
        ifelse(params_general$type_os == "Linux","2>&1", "") # crucial for Linux where LAStools seems to write to stderr
      )
    )
    file_dtm = file.path(dir_tile, paste0(name_tile,"_dtm.tif"))
    
    # sometimes, this rudimentary dtm creation seems to fail (dense point clouds, probably unsorted)
    if(!file.exists(file_dtm)){
      # try the other algorithm
      lasgorithm_dem2 = ifelse(lasgorithm_dem == "blast2dem", "las2dem", "blast2dem")
      
      return_system = system(
        paste(
          file.path.system(type_os = params_general$type_os, params_general$path_lastools, update.command_lastools(lasgorithm_dem2, type_architecture = params_general$type_architecture)),
          "-i", file.path.system(type_os = params_general$type_os, tile_pointcloud),
          "-cores 1",
          # "-kill", 100, # maximum size of triangle
          "-step", 1, # 1 m step size
          "-keep_class 2 8",
          "-o", file.path.system(type_os = params_general$type_os, dir_tile, paste0(name_tile,"_dtm.tif")),
          ifelse(params_general$type_os == "Linux","2>&1", "") # crucial for Linux where LAStools seems to write to stderr
        )
      )
    }
    
    if(file.size(file_dtm) == 0 | file.size(file_dtm_highest) == 0){
      # if there is not enough ground points in the tile to create a raster
      file.copy(tile_pointcloud, file.path(path_output, paste0(name_tile, ".laz")))
    } else {
      # create a vector of file.paths (additional ground files will be added on top if necessary)
      tiles_ground_system = file.path.system(type_os = params_general$type_os, tile_ground) 
      
      # now aggregate dtm to assess slope
      dtm = rast(file_dtm)
      dtm_highest = rast(file_dtm_highest)
      dtm_highest = extend(crop(dtm_highest, dtm), dtm)
      
      dtm_agg = aggregate(dtm, fact = 5, fun = "mean", na.rm = T)
      dtm_highest_agg = aggregate(dtm_highest, fact = 5, fun = "mean", na.rm = T)
      
      mat_circular = focalMat(dtm_agg, radius_dtm, type = "circle") # draw a 25m circle around each point
      mat_circular[mat_circular != 0] = 1
      
      # there are sometimes tiny tiles at the edges that can be smaller than the smoothing matrix
      # these are irrelevant for ground refinement, but can cause problems for the focal operation, so we allow for an exception in those cases
      dim_dtm_agg = min(nrow(dtm_agg),ncol(dtm_agg))
      dim_mat_circular = min(nrow(mat_circular),ncol(mat_circular))
      
      # new in v.1.0.1: this is the minimum condition so that terra does not throw an error, but it is also quite extreme (for this exception to be not fulfilled, the raster needs to be less than 25m in one direction, assuming a 25m smoothing operator)
      if(dim_mat_circular < 2 * dim_dtm_agg){
        dtm_agg = terra::focal(dtm_agg, w = mat_circular, fun = "mean", na.rm = T)
        slope_dtm = terra::terrain(dtm_agg, v = "slope", unit = "degrees", neighbors = 8)
        
        dtm_highest_agg = focal(dtm_highest_agg, w = mat_circular, fun = "mean", na.rm = T)
        slope_dtm_highest = terra::terrain(dtm_highest_agg, v = "slope", unit = "degrees", neighbors = 8)
        
        # define point around which to turn
        zmax = round(as.numeric(global(dtm, "max",na.rm = T)))
        zmin = round(as.numeric(global(dtm, "min",na.rm = T)))
        
        origin_xy = c(
          0.5 * (xmin(slope_dtm) + xmax(slope_dtm))
          , 0.5 * (ymin(slope_dtm) + ymax(slope_dtm))
        )
        origin_xz = c(
          0.5 * (xmin(slope_dtm) + xmax(slope_dtm))
          , 0.5 * (zmin + zmax)
        )
        
        # Common settings
        # Default: step is 25 m, sub is 5, bulge is 2 m, spike is 1+1 m, and offset is 0.05 m
        # Nature: step is 5 m, sub is 3, bulge is 1 m, spike is 1+1 m, and offset is 0.05 m
        # extra_fine changes sub to 7, bulge should be 1/10th of step size, but is clamped to between 1 and 2 by default
        # Here we use highest resolution for the initial reclassification and slightly lower resolution when rotating the point cloud
        
        steps_refinement = data.table(
          minangle_steep = c(40, 60)
          , maxangle_xz = c(20, 30)
          , nbincrements_xz = c(1, 1)
          , minradius_NA = c(15, NA)
          , option_initial = c("-bulge 3.0 -hyper_fine","-nature -extra_fine")
          , option_rotated = c("-bulge 3.0 -extra_fine","-nature")
        )
        
        # define intermediate files
        tile_pointcloud_refined = file.path(dir_tile, paste0(name_tile,"_refined.laz"))
        tile_pointcloud_rotated = file.path(dir_tile, paste0(name_tile,"_rotated.laz"))
        tile_pointcloud_tilted = file.path(dir_tile, paste0(name_tile,"_tilted.laz"))
        tile_pointcloud_tiltedback = file.path(dir_tile, paste0(name_tile,"_tiltedback.laz"))
        
        for(i in 1:nrow(steps_refinement)){
          # get analysis parameters
          minangle_steep = steps_refinement[i]$minangle_steep
          maxangle_xz = steps_refinement[i]$maxangle_xz
          nbincrements_xz = steps_refinement[i]$nbincrements_xz
          minradius_NA = steps_refinement[i]$minradius_NA
          option_initial = steps_refinement[i]$option_initial
          option_rotated = steps_refinement[i]$option_rotated
          
          # get steep areas
          areas_steep = ifel(slope_dtm >= minangle_steep | slope_dtm_highest >= minangle_steep, 1, NA)
          
          # add NA areas
          if(!is.na(minradius_NA)){
            # get nona
            dtm_nona_agg = aggregate(ifel(is.na(dtm_highest),NA,1), fact = 5, fun = "mean", na.rm = T)
            mat_circular = focalMat(dtm_nona_agg, minradius_NA, type = "circle") # draw a circle around each point
            mat_circular[mat_circular != 0] = 1
            dtm_nona_agg = focal(dtm_nona_agg, w = mat_circular, fun = "mean", na.rm = T)
            dtm_na_agg = focal(ifel(is.na(dtm_nona_agg),1,NA), w = mat_circular, fun = "mean", na.rm = T)
            
            # combine with other steep areas
            areas_steep = ifel(areas_steep == 1 | dtm_na_agg == 1, 1, NA) 
          }
          
          nona_areas_steep = as.numeric(global(areas_steep, fun="notNA"))
          
          if(nona_areas_steep > 0){
            # now we rotate the point cloud in 8 cardinal directions
            # the easiest way to do this seems by rotating the entire tile 8 times (each time 45 degrees) and do a single-direction upwards/downwards flip
            # alternatives would be to rotate 4 times and do two upwards/downwards flips, etc. But then an extra rotation back or forth by 180 degrees is needed
            # technically, this segment could be taken outside of the loop, but we keep it here for now
            
            
            # we copy the file into the current directory
            if(option_initial != ""){
              # new in v.50: previously we did not reclassify in the untilted mode, although this should be the most reliable classification
              # classify the tilted point cloud
              return_system = system(
                paste(
                  file.path.system(type_os = params_general$type_os, params_general$path_lastools, update.command_lastools("lasground_new", type_architecture = params_general$type_architecture)),
                  "-i", file.path.system(type_os = params_general$type_os, tile_pointcloud),
                  "-cores 1",
                  "-non_ground_unchanged", # crucial: keeps previous classifications (trivially true for ground classifications)
                  # "-ground_class 2", # not necessary
                  option_initial,
                  "-o", file.path.system(type_os = params_general$type_os, tile_pointcloud_rotated),
                  ifelse(params_general$type_os == "Linux","2>&1", "") # crucial for Linux where LAStools seems to write to stderr
                )
              )
            } else {
              # otherwise just copy it
              file.copy(from = tile_pointcloud,to = tile_pointcloud_rotated, overwrite = T)
            }
            
            if(maxangle_xz > 0 & nbincrements_xz >= 1 & nbincrements_xy>= 1){
              angle_xy = 360/nbincrements_xy
              angle_xz = maxangle_xz/nbincrements_xz
              
              for(increment_xy in 1:nbincrements_xy){
                # first take the rotated point cloud and tilt it
                cat("Angle:",(increment_xy-1) * angle_xy,"\n")
                cat("Tilt the point cloud up to",maxangle_xz,"in",nbincrements_xz,"increments and reclassify\n")
                for(increment_xz in 1:nbincrements_xz){
                  return_system = system(
                    paste(
                      file.path.system(type_os = params_general$type_os, params_general$path_lastools, update.command_lastools("las2las", type_architecture = params_general$type_architecture)),
                      "-i", file.path.system(type_os = params_general$type_os, tile_pointcloud_rotated),
                      "-cores 1",
                      "-rotate_xz", paste0(angle_xz," ", origin_xz[1], " ", origin_xz[2]), 
                      "-o", file.path.system(type_os = params_general$type_os, tile_pointcloud_tilted),
                      ifelse(params_general$type_os == "Linux","2>&1", "") # crucial for Linux where LAStools seems to write to stderr
                    )
                  )
                  
                  # classify the tilted point cloud
                  return_system = system(
                    paste(
                      file.path.system(type_os = params_general$type_os, params_general$path_lastools, update.command_lastools("lasground_new", type_architecture = params_general$type_architecture)),
                      "-i", file.path.system(type_os = params_general$type_os, tile_pointcloud_tilted),
                      "-cores 1",
                      "-non_ground_unchanged", # crucial: keeps previous classifications (trivially true for ground classifications)
                      # "-ground_class 2", # not necessary
                      option_rotated,
                      "-o", file.path.system(type_os = params_general$type_os, tile_pointcloud_rotated),
                      ifelse(params_general$type_os == "Linux","2>&1", "") # crucial for Linux where LAStools seems to write to stderr
                    )
                  )
                }
                
                # convert back to untilted
                cat("Tilt the point cloud back by",-maxangle_xz, "\n")
                return_system = system(
                  paste(
                    file.path.system(type_os = params_general$type_os, params_general$path_lastools, update.command_lastools("las2las", type_architecture = params_general$type_architecture)),
                    "-i", file.path.system(type_os = params_general$type_os, tile_pointcloud_rotated),
                    "-cores 1",
                    "-rotate_xz", paste0(-maxangle_xz," ", origin_xz[1], " ", origin_xz[2]), 
                    "-o", file.path.system(type_os = params_general$type_os, tile_pointcloud_tiltedback),
                    ifelse(params_general$type_os == "Linux","2>&1", "") # crucial for Linux where LAStools seems to write to stderr
                  )
                )
                
                # finally we rotate in xy direction
                return_system = system(
                  paste(
                    file.path.system(type_os = params_general$type_os, params_general$path_lastools, update.command_lastools("las2las", type_architecture = params_general$type_architecture)),
                    "-i", file.path.system(type_os = params_general$type_os, tile_pointcloud_tiltedback),
                    "-cores 1",
                    "-rotate_xy", paste0(angle_xy," ", origin_xy[1], " ", origin_xy[2]), 
                    "-o", file.path.system(type_os = params_general$type_os, tile_pointcloud_rotated),
                    ifelse(params_general$type_os == "Linux","2>&1", "") # crucial for Linux where LAStools seems to write to stderr
                  )
                )
              }
            }
            
            # convert to polygons
            # we do not merge polygons, but split them by cell ids; this is IMPORTANT, as otherwise points within donut-like polygons might be extracted too
            values(areas_steep)[!is.na(values(areas_steep))] = cells(areas_steep)
            polys_steep = as.polygons(areas_steep, values = T)
            writeVector(polys_steep, filename = file.path(dir_tile, paste0(name_tile,"_steep.shp")), overwrite = T)
            
            # extract ground points
            tile_groundextra = file.path(dir_tile, paste0(name_tile,"_",i,"_groundextra.laz")) # for writing out 
            
            # this will throw lots of warnings of duplicate points due to our usage of grid cells as polygons (points close to polygon edges will probably be included in both)
            # but no issue: we run lasduplicate at the end anyways
            return_system = system(
              paste(
                file.path.system(type_os = params_general$type_os, params_general$path_lastools, update.command_lastools("lasclip", type_architecture = params_general$type_architecture)),
                "-i", file.path.system(type_os = params_general$type_os, tile_pointcloud_rotated),
                "-cores 1",
                "-poly", file.path.system(type_os = params_general$type_os, dir_tile, paste0(name_tile,"_steep.shp")),
                "-keep_class 2",
                "-o", file.path.system(type_os = params_general$type_os, tile_groundextra),
                "-dont_remove_empty_files",
                ifelse(params_general$type_os == "Linux","2>&1", "") # crucial for Linux where LAStools seems to write to stderr
              )
            )
            
            # put file paths together
            tiles_ground_system = c(tiles_ground_system, file.path.system(type_os = params_general$type_os, tile_groundextra)) 
          }
        }
      }
      
      if(length(tiles_ground_system) == 1){
        # if there are no additional ground tiles, we simply copy the original tile with the default ground classification
        file.copy(tile_pointcloud, file.path(path_output, paste0(name_tile, ".laz")))
      } else {
        # otherwise we define a new ground point file
        tiles_ground_system = paste0(tiles_ground_system, collapse = " ")
        tile_groundnew = file.path(dir_tile, paste0(name_tile,"_groundnew.laz"))
        
        return_system = system(
          paste(
            file.path.system(type_os = params_general$type_os, params_general$path_lastools, update.command_lastools("lasmerge", type_architecture = params_general$type_architecture)),
            "-i", tiles_ground_system,
            "-cores", 1,
            "-o", file.path.system(type_os = params_general$type_os, tile_groundnew),
            "-keep_lastiling",
            ifelse(params_general$type_os == "Linux","2>&1", "") # crucial for Linux where LAStools seems to write to stderr
          )
        )
        
        # now we reclassify everything within 5 cm of the new ground points as ground
        # this is better than using the rotated point clouds, because rotation could add rounding errors to points (also the VLRs are thus guaranteed to stay the same)
        # but first we remove duplicates
        return_system = system(
          paste(
            file.path.system(type_os = params_general$type_os, params_general$path_lastools, update.command_lastools("lasduplicate", type_architecture = params_general$type_architecture)),
            "-i", file.path.system(type_os = params_general$type_os, tile_groundnew),
            "-cores", 1,
            "-unique_xyz",
            "-o", file.path.system(type_os = params_general$type_os, tile_ground),
            ifelse(params_general$type_os == "Linux","2>&1", "") # crucial for Linux where LAStools seems to write to stderr
          )
        )
        
        # now update ground classification based on supplied ground points, everything gets classified as 8
        return_system = system(
          paste(
            file.path.system(type_os = params_general$type_os, params_general$path_lastools, update.command_lastools("lasheight", type_architecture = params_general$type_architecture)),
            "-i", file.path.system(type_os = params_general$type_os, tile_pointcloud),
            "-cores", 1,
            "-ground_points", file.path.system(type_os = params_general$type_os, tile_ground),
            "-classify_between -0.05 0.05 8",
            "-o", file.path.system(type_os = params_general$type_os, tile_pointcloud_refined),
            ifelse(params_general$type_os == "Linux","2>&1", "") # crucial for Linux where LAStools seems to write to stderr
          )
        )
        
        # we run a final ground classification to obtain original ground classification
        # these ground points overwrite all previous ground points with class 2, but any class 8 ground points don't get removed
        # and we write them straight to the output directory
        return_system = system(
          paste(
            file.path.system(type_os = params_general$type_os, params_general$path_lastools, update.command_lastools("lasground_new", type_architecture = params_general$type_architecture)),
            "-i", file.path.system(type_os = params_general$type_os, tile_pointcloud_refined),
            "-cores 1",
            "-non_ground_unchanged", # crucial: keeps previous classifications (trivially true for ground classifications)
            "-ground_class 2",
            "-o", file.path.system(type_os = params_general$type_os, path_output, paste0(name_tile, ".laz")),
            ifelse(params_general$type_os == "Linux","2>&1", "") # crucial for Linux where LAStools seems to write to stderr
          )
        )
      }
    }
  }
  unlink(dir_tile, recursive = T)
}

# dir_output = path_output
# tile_pointcloud = tile_pointcloud_spliced
# step = 1
# 
# index_tile = 1
reclassify.tile = function(index_tile, files_groundlowest, params_general, odix = "_refined", remove.tile_pointcloud = F){
  
  # determine neighbouring groundpoint tiles
  file_groundlowest = files_groundlowest[index_tile]
  dir_current = dirname(file_groundlowest$path_refine)
  tile_pointcloud = file_groundlowest$tile_pointcloud
  
  # will include the groundpoint tile itself, which is great
  files_groundlowest_adjacent = files_groundlowest[
    (
      xmin <= file_groundlowest$xmax & xmax >= file_groundlowest$xmin
    ) & (
      ymin <= file_groundlowest$ymax & ymax >= file_groundlowest$ymin
    )
  ]
  
  # create temporary directory to do further processing (to avoid overlap with other processes)
  tmpdir_tile = file.path(dir_current,paste0("cluster",index_tile))
  if(!dir.exists(tmpdir_tile)){dir.create(tmpdir_tile)}
  
  file.copy(files_groundlowest_adjacent$path_refine, file.path(tmpdir_tile,basename(files_groundlowest_adjacent$path_refine)))
  
  unlink(tmpdir_tile,recursive = T)
  
  return(file.path(dir_current,gsub(paste0(".",params_general$type_file),paste0(odix,".laz"),basename(tile_pointcloud), fixed = T)))
}

# filter according to height
# !!! TODO: function should be merged with second call in pit free CHM creation and combined into a proper lasheight function
# height_lim is used as a very coarse cloud filter (assuming that trees never reach 125m)
filter.height = function(params_general, path_output = "", height_lim = 125, update.path = TRUE){
  
  # define output directory
  if(path_output == ""){
    path_output = file.path(params_general$path_data,paste0("height_lim", height_lim))
  }
  if(!dir.exists(path_output)) dir.create(path_output)
  
  # pass lastools command to system
  if(!is.voidstring(params_general$path_lastools)){
    return_system = system(
      paste(
        file.path.system(type_os = params_general$type_os, params_general$path_lastools, update.command_lastools("lasheight", type_architecture = params_general$type_architecture)),
        "-i",
        file.path.system(type_os = params_general$type_os, params_general$path_data,paste0("*.",params_general$type_file)),
        "-cores", params_general$n_cores,
        "-do_not_store_in_user_data",
        "-drop_above", height_lim,
        "-odir", file.path.system(type_os = params_general$type_os, path_output),
        "-olaz",
        ifelse(params_general$type_os == "Linux","2>&1", "") # crucial for Linux where LAStools seems to write to stderr
      )
    )
    
    # lasindex(params_general)
  } else {
    cat("filter.height open source called...\n")
    
    tri <- triangulate(filter = keep_ground_and_water())
    extra <- add_extrabytes("int", "HAG", "Height Above Ground")
    trans <- transform_with(tri, store_in_attribute = "HAG")
    filter <- lasR::delete_points(paste0("HAG > ", height_lim))
    
    filter_height_pipeline =
      
      lasR::reader_las() + tri + extra + trans + filter +
      
      write_las(paste0(path_output, "/*.laz"))
    
    exec(filter_height_pipeline,
         on = params_general$path_data,
         ncores = params_general$n_cores,
         progress = TRUE)
    
    cat("filter.height open source done!\n")
  }
  
  if(params_general$cleanup == T) cleanup.files(params_general$path_data)
  
  if(update.path == TRUE){
    cat("Amending file path and reindexing\n")
    params_general$path_data = path_output
    params_general$type_file = "laz"
    cat("Amended file path returned\n")
    return(params_general)
  }
}

downsample = function(params_general, path_output = "", pd_downsampling = 2, update.path = TRUE){
  # define step size
  step_downsampling = round(sqrt(1/pd_downsampling),3)
  
  # define output directory
  if(path_output == ""){
    path_output = file.path(params_general$path_data,paste0("downsample", pd_downsampling))
  }
  if(!dir.exists(path_output)) dir.create(path_output)
  
  if(!is.voidstring(params_general$path_lastools)){
    # first we thin the data
    return_system = system(
      paste(
        file.path.system(type_os = params_general$type_os, params_general$path_lastools, update.command_lastools("lasthin", type_architecture = params_general$type_architecture)),
        "-i",
        file.path.system(type_os = params_general$type_os, params_general$path_data,paste0("*.",params_general$type_file)),
        "-cores", params_general$n_cores,
        "-keep_first", # makes this more robust to scanner energy (but also throws away a lot of extra information at low pulse densities)
        "-random",
        "-step", step_downsampling,
        # "-subcircle", step_downsampling * 0.5,
        "-odir", file.path.system(type_os = params_general$type_os, path_output),
        "-olaz",
        ifelse(params_general$type_os == "Linux","2>&1", "") # crucial for Linux where LAStools seems to write to stderr
      )
    )
  } else {
    # TODO: lidR version
  }
  
  if(params_general$cleanup == T) cleanup.files(params_general$path_data)
  
  if(update.path == TRUE){
    cat("Amending file path and reindexing\n")
    params_general$path_data = path_output
    params_general$type_file = "laz"
    # lasindex(params_general)
    cat("Amended file path returned\n")
    return(params_general)
  }
}

normalize.tile = function(file_ground, params_general, path_output){
  file_laz = basename(gsub("_ground.laz",".laz",file_ground))
  
  return_system = system(
    paste(
      file.path.system(type_os = params_general$type_os, params_general$path_lastools, update.command_lastools("lasheight", type_architecture = params_general$type_architecture)),
      "-i",
      file.path.system(type_os = params_general$type_os, params_general$path_data,file_laz),
      "-ground_points ", file.path.system(type_os = params_general$type_os, file_ground),
      "-replace_z",
      "-odir", file.path.system(type_os = params_general$type_os, path_output),
      "-olaz",
      ifelse(params_general$type_os == "Linux","2>&1", "") # crucial for Linux where LAStools seems to write to stderr
    )
  )
}

normalize.pointcloud = function(params_general, path_output = "", path_dtm = "", update.path = TRUE){
  
  # define output directory
  if(path_output == ""){
    path_output = file.path(params_general$path_data,"normalize")
  }
  if(!dir.exists(path_output)) dir.create(path_output)
  
  if(!is.voidstring(params_general$path_lastools)){
    # first normalize height
    if(path_dtm == ""){
      return_system = system(
        paste(
          file.path.system(type_os = params_general$type_os, params_general$path_lastools, update.command_lastools("lasheight", type_architecture = params_general$type_architecture)),
          "-i",
          file.path.system(type_os = params_general$type_os, params_general$path_data,paste0("*.",params_general$type_file)),
          "-cores", params_general$n_cores,
          "-replace_z",
          "-classification 2 8", # important to use the refined ground points!
          "-odir", file.path.system(type_os = params_general$type_os, path_output),
          "-olaz",
          ifelse(params_general$type_os == "Linux","2>&1", "") # crucial for Linux where LAStools seems to write to stderr
        )
      )
    } else {
      # we need to compress the dtm to laz format
      return_system = system(
        paste(
          file.path.system(type_os = params_general$type_os, params_general$path_lastools, "demzip"),
          "-i",
          file.path.system(type_os = params_general$type_os, path_dtm,"*.tif"),
          "-cores", params_general$n_cores,
          "-odir", file.path.system(type_os = params_general$type_os, path_dtm),
          "-olaz",
          "-odix _ground",
          ifelse(params_general$type_os == "Linux","2>&1", "") # crucial for Linux where LAStools seems to write to stderr
        )
      )
      
      files_ground = list.files(path_dtm,pattern = "_ground.laz", full.names = T)
      
      if(params_general$n_cores > 1){
        # process in parallel
        nbcluster = params_general$n_cores
        cat("\nNormalizing point cloud in parallel\n")
        cl = makeCluster(nbcluster, outfile = file.path(path_output,"log_normalize.txt"))
        clusterExport(cl, varlist = list("normalize.tile","file.path.system","update.command_lastools")) # export function
        clusterEvalQ(cl, {library(parallel)})
        on.exit(stopCluster(cl))  # on.exit is also triggered when an error occurs
        
        # load balancing improves processing time when there are asymmetries in tile size (e.g. edge vs. center)
        parLapplyLB(cl, files_ground, normalize.tile, params_general = params_general, path_output = path_output)
      } else {
        # non-parallel version
        for(j in 1:length(files_ground)){
          normalize.tile(file_ground = files_ground[j], params_general = params_general, path_output = path_output)
        }
      }
    }
    
    # lasindex(params_general)
  }
  
  if(params_general$cleanup == T) cleanup.files(params_general$path_data)
  
  if(update.path == TRUE){
    cat("Amending file path and reindexing\n")
    params_general$path_data = path_output
    params_general$type_file = "laz"
    cat("Amended file path returned\n")
    return(params_general)
  }
}

# make dtm/dsm
# options include: dtm/dsm/dsm_spikefree
las2dem = function(params_general, path_output = "", name_raster = "", kill, step, type_output = "tif", option = "dtm", spacing_pulses = as.numeric(NA)){
  
  # record time
  time_start = Sys.time()
  
  # choose algorithm (dsm_spikefree only works with las2dem, which is slower)
  lasgorithm_dem = "blast2dem"
  if(tolower(option) == "dsm" & !is.na(spacing_pulses)){
    option = "dsm_spikefree"
    lasgorithm_dem = "las2dem"
    spacing_pulses = round(spacing_pulses,2)
  } else if(tolower(option) == "dtm_noblast" | params_general$use.blast2dem == F){
    lasgorithm_dem = "las2dem"
  }
  
  # define output directory
  if(path_output == ""){
    if(name_raster == ""){
      path_output = file.path(params_general$path_data,option)
    } else {
      path_output = file.path(params_general$path_data,name_raster)
    }
  }
  if(!dir.exists(path_output)) dir.create(path_output)
  
  if(!is.voidstring(params_general$path_lastools)){
    
    # define additional options for lastools command (only ground points for dtm, only first returns for dsm)
    # the pulse spacing has to be provided to spike_free, ideally this is chosen in a conservative way (assume wider spacing than average), otherwise it will create quite a few spikes in badly sampled areas
    additional = NULL
    if(tolower(option) == "dtm" | tolower(option) == "dtm_noblast"){
      additional = "-keep_class 2"
    } else if(tolower(option) == "dtm_firstonly"){
      additional = "-keep_class 2 -keep_first"
    } else if(tolower(option) == "dtm_refined"){
      additional = "-keep_class 2 8"
    } else if(tolower(option) == "dsm"){
      additional = "-keep_first"
    } else if(tolower(option) == "dsm_spikefree"){
      additional = paste0("-spike_free ",3 * spacing_pulses)
    } else {
      cat("Input option",option,"is not known. Please provide either DTM or DSM")
    }
    
    # pass lastools command to system
    return_system = system(
      paste(
        file.path.system(type_os = params_general$type_os, params_general$path_lastools, update.command_lastools(lasgorithm_dem, type_architecture = params_general$type_architecture)),
        "-i",
        file.path.system(type_os = params_general$type_os, params_general$path_data,paste0("*.",params_general$type_file)),
        "-cores", params_general$n_cores,
        ifelse(params_general$bbtype == "tile","-use_tile_bb",""),
        "-kill", kill, # maximum size of triangle
        "-step", step,
        "-odir", file.path.system(type_os = params_general$type_os, path_output),
        paste0("-o ",name_raster,".",type_output),
        additional,
        ifelse(params_general$type_os == "Linux","2>&1", "") # crucial for Linux where LAStools seems to write to stderr
      )
    )
    
    if(return_system == 0){
      cat("Successful completion of system command\n")
    } else {
      # logfile_dt = data.table(path = as.character(NA), issue = paste0("WARNING! las2dem or blast2dem generated non-0 return."))
      # fwrite(logfile_dt, file = params_general$logfile, append = T)
      cat("System command could not be completed\n")
    } 
  } else {
    cat("las2dem open source called...\n")
    
    read <- lasR::reader()
    # choose different lidR options, no spikefree version available
    if(tolower(option) == "dtm"){
      cat(">> dtm tin\n")
      
      tri_dtm  <- lasR::triangulate(filter = keep_ground(), max_edge = kill)
      dtm_stage  <- lasR::rasterize(step, tri_dtm, ofile = file.path(path_output, "dtm.tif"))
      pipeline <- read + tri_dtm + dtm_stage
      dtm = lasR::exec(pipeline,
                       on = params_general$path_data,
                       ncores = params_general$n_cores,
                       progress = T)
      # terra::writeRaster(dtm[[1]], file.path(path_output, "dtm.tif"), filetype = "GTiff", overwrite = TRUE)
      
    } else if(tolower(option) == "dsm"){
      
      tri_dsm  <- lasR::triangulate(filter = keep_first(), max_edge = kill)
      dsm_stage  <- lasR::rasterize(step, tri_dsm, ofile = file.path(path_output, "dsm_tin.tif")) # input is a triangulation stage
      pipeline <- read + tri_dsm + dsm_stage
      dsm = lasR::exec(pipeline,
                       on = params_general$path_data,
                       ncores = params_general$n_cores,
                       with = list(chunk = size_tile),
                       progress = T)
      # terra::writeRaster(dsm[[1]], file.path(path_output, "dsm_tin.tif"), filetype = "GTiff", overwrite = TRUE)
      
    } else if(tolower(option) == "dsm_spikefree"){
      
      print("Open source spikefree")
      
      pipeline = lasR::spikefree(res = step, filter = "ReturnNumber == 1", ofile = file.path(path_output, "dsm_spikefree.tif"))
      dsm = lasR::exec(pipeline,
                       on = params_general$path_data,
                       ncores = params_general$n_cores,
                       with = list(chunk = size_tile),
                       progress = T)
    }
    
    else {
      cat("Input option", option, "is not known. Please provide either DTM or DSM")
    }
    cat("las2dem open source done!\n")
  }
  
  time_end = Sys.time()
  time_processing = difftime(time_end,time_start,units = "mins")
  return(time_processing)
}

make.dtm_nooverhangs = function(params_general, path_output = "", path_products = "", name_raster = "dtm_refined", kill = 200, step = 1, type_output = "tif", classes_ground = "2 8", threshold_drop = 10, arguments_additional = ""){
  
  # start recording time
  time_start = Sys.time()
  
  # define output directory
  if(path_output == ""){
    path_output = file.path(params_general$path_data,name_raster)
  }
  if(!dir.exists(path_output)) dir.create(path_output)
  
  if(!is.voidstring(params_general$path_lastools))
  {
    # process in parallel
    nbcluster = params_general$n_cores
    cl = makeCluster(nbcluster)
    clusterExport(cl, varlist = list("make.dtm_nooverhangs_tile","file.path.system","update.command_lastools")) # export function, v.42 added buffer_withsf
    clusterEvalQ(
      cl,
      {
        library(data.table) # for %like% operator etc.
        library(parallel)
        library(terra)
        library(lidR)
      }
    )
    on.exit(stopCluster(cl))  # on.exit is also triggered when an error occurs
    
    tiles_pointcloud = list.files(params_general$path_data, pattern = paste0(".",params_general$type_file), full.names = T)
    
    parLapplyLB(cl, tiles_pointcloud, make.dtm_nooverhangs_tile, params_general = params_general, path_output = path_output, kill = kill, step = step, type_output = type_output, classes_ground = classes_ground, threshold_drop = threshold_drop, arguments_additional = arguments_additional)
    
    files_dtm_supplied = list.files.nonzero(path = path_output, pattern = ".tif", full.names = TRUE)
    dtm_supplied = vrt(files_dtm_supplied); terra::crs(dtm_supplied) = crs_scan
    writeRaster(dtm_supplied, filename = file.path(path_products, paste0(name_raster,addendum_name,".tif")), overwrite = T)
  }
  else
  {
    cat("\nGenerate dtm_nooverhangs open source called...\n")
    tiles_pointcloud = list.files(params_general$path_data, pattern = paste0(".",params_general$type_file), full.names = T)
    
    # TODO: replace with JR's
    
    ctg <- readLAScatalog(tiles_pointcloud)
    opt_chunk_size(ctg) <- size_tile
    opt_chunk_buffer(ctg) <- size_buffer
    opt_progress(ctg) <- T
    opt_output_files(ctg) <- file.path(params_general$path_data, paste0(name_raster, "/{ID}_dtm_nooverhangs"))
    
    future::plan(future::multisession, workers = params_general$n_cores)
    
    cat("lidR::rasterize_terrain()")
    dtm <- rasterize_terrain(ctg, res = resolution, algorithm = tin())
    dtm_tiles <- list.files(file.path(params_general$path_data, name_raster), pattern = "_dtm_nooverhangs\\.tif$", full.names = TRUE)
    dtm = terra::vrt(dtm_tiles)
    terra::writeRaster(dtm,
                       filename = file.path(path_products, paste0(name_raster, ".tif")),
                       overwrite = TRUE)
    
    future::plan(future::sequential)
    if(params_general$cleanup == T) file.remove(dtm_tiles)
    
    #params_general$path_data = path_output
    params_general$type_file = "laz"
    cat("\nGenerate dtm_nooverhangs open source done!\n")
  }
  
  time_end = Sys.time()
  time_processing = difftime(time_end,time_start,units = "mins")
  return(time_processing)
}

make.dtm_nooverhangs_tile = function(tile_pointcloud, params_general, path_output, kill = 200, step = 1, type_output = "tif", classes_ground = "2 8", threshold_drop = 10, arguments_additional = ""){
  
  # create a directory for processing
  name_tile = gsub(paste0(".",params_general$type_file),"", basename(tile_pointcloud), fixed = T)
  dir_tile = file.path(path_output, name_tile)
  if(!dir.exists(dir_tile)) dir.create(dir_tile)
  
  # thin the point cloud first (highest ground points only)
  tile_groundhighest = file.path(dir_tile, paste0(name_tile,"_groundhighest.laz"))
  
  return_system = system(
    paste(
      file.path.system(type_os = params_general$type_os, params_general$path_lastools, update.command_lastools("lasthin", type_architecture = params_general$type_architecture)),
      "-i", file.path.system(type_os = params_general$type_os, tile_pointcloud),
      "-cores", 1,
      "-highest",
      "-step", step,
      "-subcircle",step * 1.05,
      "-keep_class", classes_ground,
      "-o", file.path.system(type_os = params_general$type_os, tile_groundhighest),
      ifelse(params_general$type_os == "Linux","2>&1", "") # crucial for Linux where LAStools seems to write to stderr
    )
  )
  
  # now use these ground points to drop any points below in the original point cloud
  tile_groundrefined = file.path(dir_tile, paste0(name_tile,"_groundrefined.laz"))
  
  return_system = system(
    paste(
      file.path.system(type_os = params_general$type_os, params_general$path_lastools, update.command_lastools("lasheight", type_architecture = params_general$type_architecture)),
      "-i", file.path.system(type_os = params_general$type_os, tile_pointcloud),
      "-ground_points", file.path.system(type_os = params_general$type_os, tile_groundhighest),
      "-cores", 1,
      "-drop_below", -threshold_drop,
      "-do_not_store_in_user_data",
      "-keep_class", classes_ground,
      "-o", file.path.system(type_os = params_general$type_os, tile_groundrefined),
      ifelse(params_general$type_os == "Linux","2>&1", "") # crucial for Linux where LAStools seems to write to stderr
    )
  )
  
  # define dem algorithm
  lasgorithm_dem = ifelse(params_general$use.blast2dem == T, "blast2dem", "las2dem")
  tile_dtm = file.path(path_output, paste0(name_tile,".",type_output))
  
  # pass lastools command to system
  return_system = system(
    paste(
      file.path.system(type_os = params_general$type_os, params_general$path_lastools, update.command_lastools(lasgorithm_dem, type_architecture = params_general$type_architecture)),
      "-i", file.path.system(type_os = params_general$type_os, tile_groundrefined),
      "-cores", 1,
      ifelse(params_general$bbtype == "tile","-use_tile_bb",""),
      "-kill", kill, # maximum size of triangle
      "-step", step,
      "-o", file.path.system(type_os = params_general$type_os, tile_dtm),
      arguments_additional,
      ifelse(params_general$type_os == "Linux","2>&1", "") # crucial for Linux where LAStools seems to write to stderr
    )
  )
  unlink(dir_tile, recursive = T)
}

# create custom dtms
make.dtm_custom = function(params_general, path_output = "", name_raster, kill = 200, step, arguments_additional = ""){
  time_start = Sys.time()
  
  cat("\nGenerate make.dtm_custom open source called...\n")
  # define output directory
  if(path_output == ""){
    path_output = file.path(params_general$path_data,"dtm_custom")
  }
  if(!dir.exists(path_output)) dir.create(path_output)
  
  # now we need to reclassify
  cat("\nGround classification\n")
  params_general$cleanup = F
  params_general = lasground_new(params_general = params_general, path_output = path_output, update.path = T, step = 0.5, sub = NULL, arguments_additional = arguments_additional)
  
  params_general$cleanup = T
  las2dem(params_general = params_general, path_output = path_output, name_raster = "dtm", kill = 200, step = step, type_output = "tif", option = "dtm")
  time_end = Sys.time()
  time_processing = difftime(time_end,time_start,units = "mins")
  
  cat("\nGenerate make.dtm_custom open source done!\n")
  
  return(time_processing)
}

# new in v.48: also allow for the creation of a dtm from the "highest" ground points
# this DTM will be far from complete, with many gaps in ground classification due to trees/buildings, etc. 
# the purpose of the DTM is to provide a look at the "raw" ground data (where are zones with very few ground-classified pixels and what should be the highest elevation in them) and to correct TIN-interpolated DTMs in steep areas of the DTM (TIN-interpolation might strongly reduce elevation along these slopes and lead to artificially high areas in some CHMs such as chm_highest)
make.dtm_highest = function(params_general, path_output = "", name_raster = "dtm_highest", type_output = "tif", step = 1, subcircle = 1.0, classes = "2 8"){
  # record time
  time_start = Sys.time()
  
  # define output directory
  if(path_output == ""){
    path_output = file.path(params_general$path_data,name_raster)
  }
  if(!dir.exists(path_output)) dir.create(path_output)
  
  if(!is.voidstring(params_general$path_lastools)){
    # create a simple chm, based on highest returns
    return_system = system(
      paste(
        file.path.system(type_os = params_general$type_os, params_general$path_lastools, update.command_lastools("lasgrid", type_architecture = params_general$type_architecture)),
        "-i", file.path.system(type_os = params_general$type_os, params_general$path_data,"*.laz"),
        "-cores", params_general$n_cores,
        "-step", step,
        "-keep_class",classes, 
        "-highest",
        "-subcircle",subcircle,
        ifelse(params_general$bbtype == "tile","-use_tile_bb",""),
        "-odir", file.path.system(type_os = params_general$type_os, path_output),
        "-otif",
        ifelse(params_general$type_os == "Linux","2>&1", "") # crucial for Linux where LAStools seems to write to stderr
      )
    )
  }
  else
  {
    cat("\nGenerate make.dtm_highest open source called...\n")
    tiles_pointcloud = list.files(params_general$path_data, pattern = paste0(".",params_general$type_file), full.names = T)
    files_names.input = list.files(params_general$path_data, pattern = paste0(".",params_general$type_file), full.names = F)
    
    ctg <- readLAScatalog(tiles_pointcloud)
    opt_output_files(ctg) <- file.path(path_output, "Chunk_{ID}_dtm_highest")
    opt_chunk_size(ctg) <- size_tile
    opt_chunk_buffer(ctg) <- size_buffer
    opt_filter(ctg) <- "-keep_class 2"
    opt_progress(ctg) <- T
    
    future::plan(future::multisession, workers = params_general$n_cores)
    
    zmax_catalog <- catalog_apply(ctg, function(las_chunk) {
      r <- grid_metrics(las_chunk, ~max(Z, na.rm = TRUE), res = resolution)
      rm(las_chunk)
      gc()
      return(r)
    })
    
    dtm_highest_files = list.files(path = path_output, pattern="dtm_highest.tif", full.names = TRUE)
    dtm_highest = terra::vrt(dtm_highest_files)
    raster::writeRaster(dtm_highest, file.path(path_output, "lidR_dtm_highest.tif"), overwrite = TRUE)
    
    future::plan(future::sequential)
    
    if (params_general$cleanup == T) file.remove(dtm_highest_files)
    
    #params_general$path_data = path_output
    params_general$type_file = "laz"
    cat("\nGenerate make.dtm_highest open source done!\n")
  }
  
  time_end = Sys.time()
  time_processing = difftime(time_end,time_start,units = "mins")
  return(time_processing)
}

make.dsm_highest = function(params_general, path_output = "", name_raster = "dsm_highest", type_output = "tif", step = 1, subcircle = 0.1){
  # record time
  time_start = Sys.time()
  
  # define output directory
  if(path_output == ""){
    path_output = file.path(params_general$path_data,name_raster)
  }
  if(!dir.exists(path_output)) dir.create(path_output)
  
  if(!is.voidstring(params_general$path_lastools)){
    # create a simple chm, based on highest returns
    return_system = system(
      paste(
        file.path.system(type_os = params_general$type_os, params_general$path_lastools, update.command_lastools("lasgrid", type_architecture = params_general$type_architecture)),
        "-i", file.path.system(type_os = params_general$type_os, params_general$path_data,"*.laz"),
        "-cores", params_general$n_cores,
        "-step", step,
        "-highest",
        "-subcircle",subcircle,
        ifelse(params_general$bbtype == "tile","-use_tile_bb",""),
        "-odir", file.path.system(type_os = params_general$type_os, path_output),
        "-otif",
        ifelse(params_general$type_os == "Linux","2>&1", "") # crucial for Linux where LAStools seems to write to stderr
      )
    )
  }
  else
  {
    cat("\nGenerate make.dsm_highest lidR version called...\n")
    
    tiles_pointcloud <- list.files(params_general$path_data, pattern = paste0("\\.", params_general$type_file, "$"), full.names = TRUE)
    path_output <- path_output
    size_tile <- size_tile
    resolution <- resolution
    
    ctg <- readLAScatalog(tiles_pointcloud)
    opt_output_files(ctg) <- file.path(path_output, "Chunk_{ID}_dsm_highest")
    opt_chunk_size(ctg) <- size_tile
    opt_chunk_buffer(ctg) <- size_buffer
    opt_progress(ctg) <- TRUE
    opt_filter(ctg) <- "-keep_first"
    opt_laz_compression(ctg) <- TRUE
    
    future::plan(future::multisession, workers = params_general$n_cores)
    
    make_dsm_highest <- function(las_chunk, ...) {
      las <- readLAS(las_chunk)
      if (is.empty(las)) return(NULL)
      
      r <- grid_metrics(las, ~max(Z, na.rm = TRUE), res = resolution)
      
      rm(las_chunk, las)
      gc()
      return(r)
    }
    
    catalog_apply(ctg, make_dsm_highest)
    
    future::plan(future::sequential)
    gc()    
    
    #params_general$path_data = path_output
    params_general$type_file = "laz"
    cat("\nGenerate make.dsm_highest open source done!\n")
  }
  
  time_end = Sys.time()
  time_processing = difftime(time_end,time_start,units = "mins")
  return(time_processing)
}

make.chm_pitfree = function(params_general, path_output = "", name_raster = "chm_pitfree", type_output = "tif", path_chm_highest = "", step = 1, thinning = 0.5, kill = 3, subcircle = 0.1, resolution_pitfree = 5){
  
  # record time
  time_start = Sys.time()
  
  # define dem algorithm
  lasgorithm_dem = ifelse(params_general$use.blast2dem == T, "blast2dem", "las2dem")
  
  # define output directory
  if(path_output == ""){
    path_output = file.path(params_general$path_data,name_raster)
  }
  if(!dir.exists(path_output)) dir.create(path_output)
  
  if(!is.voidstring(params_general$path_lastools)){
    tmp_processing = file.path(path_output,"tmp_processing")
    if(!dir.exists(tmp_processing))dir.create(tmp_processing)
    
    # first we thin the data
    return_system = system(
      paste(
        file.path.system(type_os = params_general$type_os, params_general$path_lastools, update.command_lastools("lasthin", type_architecture = params_general$type_architecture)),
        "-i",
        file.path.system(type_os = params_general$type_os, params_general$path_data,"*.laz"),
        "-cores", params_general$n_cores,
        "-highest",
        "-subcircle",subcircle,
        "-step", thinning, # usually half-step!
        "-odir", file.path.system(type_os = params_general$type_os, tmp_processing),
        "-olaz",
        ifelse(params_general$type_os == "Linux","2>&1", "") # crucial for Linux where LAStools seems to write to stderr
      )
    )
    
    params_general$path_data = tmp_processing
    
    # lasindex(params_general)
    
    # create a simple chm, based on highest returns
    return_system = system(
      paste(
        file.path.system(type_os = params_general$type_os, params_general$path_lastools, update.command_lastools("lasgrid", type_architecture = params_general$type_architecture)),
        "-i",
        file.path.system(type_os = params_general$type_os, tmp_processing,"*.laz"),
        "-cores", params_general$n_cores,
        "-step", step,
        "-highest",
        ifelse(params_general$bbtype == "tile","-use_tile_bb",""),
        "-odir", file.path.system(type_os = params_general$type_os, tmp_processing),
        "-otif",
        "-odix _chm_highest",
        ifelse(params_general$type_os == "Linux","2>&1", "") # crucial for Linux where LAStools seems to write to stderr
      )
    )
    
    # read the chm back in and determine the highest point
    files_chm = list.files(path = tmp_processing, pattern = "_chm_highest.tif", full.names = TRUE)
    files_thinned = list.files(path = tmp_processing, pattern = ".laz", full.names = TRUE)
    
    # now pitfree chm
    # subset the files to process, because sometimes border areas contain only buffer points
    # this creates empty .bil files (or .tif files with blast2dem), but no associated .hdr files, throwing an error in assembly of the pitfree chm
    max_chm_bytile = rbindlist(lapply(files_chm, get.max_raster))
    max_chm_bytile[, file_thinned := gsub("_chm_highest.tif",".laz", file_raster, fixed = T)]
    max_chm_bytile = max_chm_bytile[max_raster > -Inf & max_raster < Inf] # remove problematic tiles
    max_chm = max(max_chm_bytile$max_raster, na.rm = T) + 1
    
    # create a list of all files to process
    files_nona = max_chm_bytile[,c("file_thinned")]
    path_lof_nona = file.path(tmp_processing,paste0("lof_nona.txt"))
    fwrite(files_nona,file = path_lof_nona,row.names = F, col.names = F)
    
    # create a temporary dtm
    return_system = system(
      paste(
        file.path.system(type_os = params_general$type_os, params_general$path_lastools, update.command_lastools(lasgorithm_dem, type_architecture = params_general$type_architecture)),
        "-lof",
        file.path.system(type_os = params_general$type_os, path_lof_nona),
        # "-i",
        # file.path.system(type_os = params_general$type_os, tmp_processing,"*_normalized.laz"),
        "-cores", params_general$n_cores,
        "-drop_z_above",0.1,
        ifelse(params_general$bbtype == "tile","-use_tile_bb",""),
        "-step", step,
        "-odir", file.path.system(type_os = params_general$type_os, tmp_processing),
        "-obil",
        "-odix _ground",
        ifelse(params_general$type_os == "Linux","2>&1", "") # crucial for Linux where LAStools seems to write to stderr
      )
    )
    
    # determine layers to go through for pitfree algorithm
    layers_pitfree = seq(0,max_chm,resolution_pitfree)
    
    for(z in layers_pitfree){
      
      # filter out all files whose maximum is below the current pitfree threshold + resolution_pitfree (one buffering layer) and who do not have enough point (100) above that threshold
      # this is only to improve speed, as they will simply return NA when all their points are dropped
      # z = 0
      files_above_z = max_chm_bytile[z <= (max_raster - resolution_pitfree) & z <= first100,c("file_thinned")]
      
      if(nrow(files_above_z) > 0){
        path_lof_z = file.path(tmp_processing,paste0("lof_",z,".txt"))
        fwrite(files_above_z,file = path_lof_z,row.names = F, col.names = F)
        
        return_system = system(
          paste(
            file.path.system(type_os = params_general$type_os, params_general$path_lastools, update.command_lastools(lasgorithm_dem, type_architecture = params_general$type_architecture)),
            "-lof",
            file.path.system(type_os = params_general$type_os, path_lof_z),
            # "-i",
            # file.path.system(type_os = params_general$type_os, tmp_processing,"*_thinned.laz"),
            "-cores", params_general$n_cores,
            "-kill", kill,
            "-step", step,
            "-drop_z_below", z,
            ifelse(params_general$bbtype == "tile","-use_tile_bb",""),
            "-odir", file.path.system(type_os = params_general$type_os, tmp_processing),
            "-obil",
            "-odix", paste0("chm_",sprintf("%03d", as.numeric(z))),
            ifelse(params_general$type_os == "Linux","2>&1", "") # crucial for Linux where LAStools seems to write to stderr
          )
        )
      }
    }
    
    # the following can be done beforehand with max_chm_bytile, so not necessary anymore
    # remove empty .bil files (sometimes created on boundaries, once the buffers are removed)
    # bilfiles = list.files(tmp_processing, pattern = ".bil", full.names = T)
    # bilfiles_sizes = rbindlist(lapply(bilfiles, function(x){filesize = file.size(x);y = data.table(file = x, filesize = filesize);return(y)}))
    # files.removed = file.remove(bilfiles_sizes[filesize == 0]$file)
    
    # put it all together
    return_system = system(
      paste(
        file.path.system(type_os = params_general$type_os, params_general$path_lastools, update.command_lastools("lasgrid", type_architecture = params_general$type_architecture)),
        "-i",
        file.path.system(type_os = params_general$type_os, tmp_processing,"*.bil"),
        #"-cores", params_general$n_cores,
        #paste0("-use_", params_general$bbtype, "_bb"),
        "-merged",
        "-step", step, # needed?
        "-highest",
        "-use_bb",     # needed?
        "-odir", file.path.system(type_os = params_general$type_os, path_output),
        paste0("-o ",name_raster,".",type_output),
        ifelse(params_general$type_os == "Linux","2>&1", "") # crucial for Linux where LAStools seems to write to stderr
      )
    )
    
    if(params_general$cleanup == T) cleanup.files(tmp_processing)
    
  } else {
    # first normalize height
    files.input = list.files(path = params_general$path_data,pattern = paste0("\\.",params_general$type_file), full.names = TRUE)
    ctg = readLAScatalog(files.input)
    
    opt_filter(ctg) = "-drop_withheld"
    opt_chunk_buffer(ctg) = params_general$buffer
    opt_output_files(ctg) = paste0(path_output,"/{*}")
    opt_progress(ctg) = FALSE
    
    normalize_height(ctg, algorithm = tin(), res = step) # no kill option
    
    # rasterize canopy
    files.input.normalized = list.files(path = params_general$path_data,pattern = paste0("\\.",params_general$type_file), full.names = TRUE)
    ctg.normalized = readLAScatalog(files.input.normalized)
    
    opt_filter(ctg.normalized) = "-drop_withheld"
    opt_chunk_buffer(ctg.normalized) = params_general$buffer
    opt_output_files(ctg.normalized) = paste0(path_output,"/{*}")
    opt_progress(ctg.normalized) = FALSE # deactivate rendering of progress
    rasterize_canopy(ctg.normalized,res = step, algorithm = pitfree(max_edge = kill, thresholds = layers_pitfree, subcircle = subcircle))
  }
  
  time_end = Sys.time()
  time_processing = difftime(time_end,time_start,units = "mins")
  return(time_processing)
}

make.dsm_spikefree_adaptive = function(tile_pointcloud, params_general, dir_processing, fact_agg = 5, buffer_contours = 10, step = 1, kill = 20, multi_freeze = 3.1,slope_freeze = 1.75, offset_freeze = 2.1, path_pdagg = "", perturbation_max = 0.1, timeout_lspikefree_max = 600, step_thinning = 0.1){
  
  cat("Processing", basename(tile_pointcloud),"\n")
  
  # now we write it out to the original directory
  file_dsm_spikefree_tile = file.path(dir_processing,gsub(".laz",".tif",basename(tile_pointcloud)))
  
  if(!file.exists(file_dsm_spikefree_tile))
  {
    # new in v.42, used to tackle problem with high-density point clouds
    # cf. https://groups.google.com/g/lastools/c/oY2gcW6w7HY/m/37zx06aJAQAJ
    factor_scale = NULL
    if(perturbation_max > 0){
      header_las = readLASheader(tile_pointcloud)
      factor_scale = max(c(header_las$`X scale factor`,header_las$`Y scale factor`)) # we only use x/y dimension for perturbation
      cat("Using maximum perturbation of", perturbation_max, "with a scale factor of",factor_scale,"to speed up las2dem\n")
    }
    
    # convert to standardized scale (1m)
    fact_agg = round(fact_agg * 1.0/step)
    
    # explicitly cast as numeric
    multi_freeze = as.numeric(multi_freeze)
    slope_freeze = as.numeric(slope_freeze)
    offset_freeze = as.numeric(offset_freeze)
    
    # make folder to process in (best to separate)
    dir_tile = file.path(dir_processing,gsub(".laz","",basename(tile_pointcloud)))
    if(!dir.exists(dir_tile)) dir.create(dir_tile, recursive = T)
    
    if(step_thinning > 0){
      # new in v.50, thinning to remove large point densities
      return_system = system(
        paste(
          file.path.system(type_os = params_general$type_os, params_general$path_lastools, update.command_lastools("lasthin", type_architecture = params_general$type_architecture)),
          "-i", file.path.system(type_os = params_general$type_os, tile_pointcloud),
          "-cores", 1,
          "-random",
          "-keep_first",
          "-step", step_thinning,
          "-odir", file.path.system(type_os = params_general$type_os, dir_tile),
          "-olaz",
          ifelse(params_general$type_os == "Linux","2>&1", "") # crucial for Linux where LAStools seems to write to stderr
        )
      )
    } else {
      file.copy(tile_pointcloud, dir_tile)
    }
    
    # new in v.50, create pd raster and remove NA values inside the raster (to obtain correct pulse density)
    return_system = system(
      paste(
        file.path.system(type_os = params_general$type_os, params_general$path_lastools, update.command_lastools("lasgrid", type_architecture = params_general$type_architecture)),
        "-i", file.path.system(type_os = params_general$type_os, dir_tile, basename(tile_pointcloud)),
        ifelse(params_general$bbtype == "tile","-use_tile_bb",""),
        ifelse(step > 5, "-point_density_32bit","-point_density_16bit"),
        "-cores", 1,
        "-keep_first",
        "-step", step,
        "-odir", file.path.system(type_os = params_general$type_os, dir_tile),
        "-otif",
        ifelse(params_general$type_os == "Linux","2>&1", "") # crucial for Linux where LAStools seems to write to stderr
      )
    )
    
    pd = rast(file.path(dir_tile, gsub(".laz",".tif",basename(tile_pointcloud))))
    names(pd) = "pd"
    
    nbnona = as.numeric(global(pd,"notNA"))
    
    if(nbnona > 0){
      mask_pd = as.polygons(clamp(pd,lower = 0, upper = 0, values = T), dissolve = T, values = T)
      mask_pd = fillHoles(mask_pd)
      mask_pd = rasterize(mask_pd,pd, "pd")
      
      pd_nona = sum(pd, mask_pd, na.rm = T)
      pd_agg = aggregate(pd_nona,fact = fact_agg, fun = "mean",na.rm = T)
      
      # now create separate pd levels (layers of 0.5 until 3, then 1 until 10, then 2 until 20, then 5 until 50)
      pd_agg_levels = clamp(pd_agg, lower = 0.5, values = T)
      pd_agg_levels = ifel(pd_agg_levels < 3, floor(pd_agg_levels*2)/2, pd_agg_levels)
      pd_agg_levels = ifel(pd_agg_levels >= 3 & pd_agg_levels < 10, floor(pd_agg_levels), pd_agg_levels)
      pd_agg_levels = ifel(pd_agg_levels >= 10 & pd_agg_levels < 20, floor(pd_agg_levels/2) * 2, pd_agg_levels)
      pd_agg_levels = ifel(pd_agg_levels >= 20 & pd_agg_levels < 50, floor(pd_agg_levels/5) * 5, pd_agg_levels)
      pd_agg_levels = ifel(pd_agg_levels >= 50 & pd_agg_levels < 100, floor(pd_agg_levels/25) * 25, pd_agg_levels)
      pd_agg_levels = clamp(pd_agg_levels, upper = 100, values = T)
      
      # get pd clusters and process them one by one
      contours_pd = as.polygons(pd_agg_levels, values = T, trunc = F) # very important: trunc needs to be FALSE, otherwise integers are returned
      contours_pd$id = paste0(as.integer(contours_pd$pd),"dot",round((contours_pd$pd - as.integer(contours_pd$pd)) * 100)) # lastools does not seem to like actual dots in contour layers
      writeVector(contours_pd, filename = file.path(dir_tile, "contours_pd.shp"), overwrite = T)
      contours_pd = vect(file.path(dir_tile, "contours_pd.shp"))
      contours_pd_buffered = buffer_withsf(contours_pd, buffer_contours)
      contours_pd_buffered = fillHoles(contours_pd_buffered)
      
      # write to file
      path_contours_pd_buffered = file.path(dir_tile, "contours_pd_buffered.shp")
      writeVector(contours_pd_buffered, filename = path_contours_pd_buffered, overwrite = T)
      
      return_system = system(
        paste(
          file.path.system(type_os = params_general$type_os, params_general$path_lastools, update.command_lastools("lasclip", type_architecture = params_general$type_architecture)),
          "-i", file.path.system(type_os = params_general$type_os, dir_tile, basename(tile_pointcloud)),
          "-cores", 1,
          "-keep_first", # new in v.42, remove all non-first times before processing starts (quicker and cleaner)
          "-odir", file.path.system(type_os = params_general$type_os, dir_tile),
          "-olaz",
          "-poly", file.path.system(type_os = params_general$type_os, path_contours_pd_buffered),
          "-split id",
          ifelse(params_general$type_os == "Linux","2>&1", "") # crucial for Linux where LAStools seems to write to stderr
        )
      )
      
      # create an empty raster with same resolution as pd, but NA values
      dsm_spikefree = pd
      values(dsm_spikefree) = NA
      
      timeout_lspikefree_min = 60
      if(timeout_lspikefree_max < timeout_lspikefree_min) timeout_lspikefree_max = timeout_lspikefree_min * 2
      timeout_lspikefree = timeout_lspikefree_max # default 
      diff_time_endtotal = 0
      # i = 1
      # i = 12
      for(i in 1:nrow(contours_pd)){
        cat("Creating spikefree DSM for level",i,"\n")
        
        # first determine pd level and select the current polygons
        contours_pd_current = contours_pd[i]
        pd_current = contours_pd_current$pd
        id_current = contours_pd_current$id
        
        dist_freeze = as.numeric(NA)
        if(offset_freeze <= -1){
          dist_freeze = 3 * 1/sqrt(pd_current)
        } else {
          argument_log = slope_freeze * pd_current + offset_freeze
          if(argument_log < 1.5){
            # change computation when argument approaches 1, e.g. log approaches 0, and 1/log(arg) approaches Inf
            argument_log = 1 + 0.5 * (argument_log/1.5)^1.5 # at 1.5, this recuperates the limiting case
          }
          
          dist_freeze = multi_freeze / log(argument_log) + 1.0
          #dist_freeze = multi_freeze * exp(-slope_freeze * pd_current) + offset_freeze
        }
        dist_freeze = round(dist_freeze,2)
        
        time_start = Sys.time()
        
        # new in v.42: either use sorting or optimize to accelerate spike-free processing and prevent hanging of pipeline for odd point clouds
        # cf. also https://groups.google.com/g/lastools/c/1-9YNG8FfSM/m/M71yIsIzCQAJ
        # and (unanswered): https://groups.google.com/g/lastools/c/LTc0v7GLCPk/m/3BhrDpb-BAAJ
        
        output_file = file.path.system(type_os = params_general$type_os, dir_tile, paste0(contours_pd_current$id,"_sort.laz"))
        cat(output_file)
        
        if(!file.exists(output_file))
        {
          return_system = system(
            paste(
              file.path.system(type_os = params_general$type_os, params_general$path_lastools, update.command_lastools("lasoptimize", type_architecture = params_general$type_architecture)),
              "-i", file.path.system(type_os = params_general$type_os, dir_tile, paste0(contours_pd_current$id,".laz")),
              "-cores", 1,
              "-odir", file.path.system(type_os = params_general$type_os, dir_tile),
              "-olaz",
              "-odix _sort",
              ifelse(params_general$type_os == "Linux","2>&1", "") # crucial for Linux where LAStools seems to write to stderr
            )
          )
        }
        
        # it has happened (in a G-LiHT scan) that lasoptimize failed (ERROR: x and y are not inside of lasquadtree)
        # to avoid that the code breaks, we simply copy the file onto a new file with the name _sort
        # everything else can then proceed as normal
        if(!file.exists(file.path(dir_tile, paste0(contours_pd_current$id,"_sort.laz")))){
          file.copy(file.path(dir_tile, paste0(contours_pd_current$id,".laz")),file.path(dir_tile, paste0(contours_pd_current$id,"_sort.laz")),overwrite = T)
        }
        
        # calculate spikefree
        if(!file.exists(file.path.system(type_os = params_general$type_os, dir_tile, paste0(contours_pd_current$id,"_sort.tif"))))
        {
          return_system = system(
            paste(
              file.path.system(type_os = params_general$type_os, params_general$path_lastools, update.command_lastools("las2dem", type_architecture = params_general$type_architecture)),
              "-i", file.path.system(type_os = params_general$type_os, dir_tile, paste0(contours_pd_current$id,"_sort.laz")), # new in v.42
              "-cores", 1,
              ifelse(params_general$bbtype == "tile","-use_tile_bb",""),
              # "-keep_first", # removed in v.42 and moved to splitting procedure
              "-kill", kill, # maximum size of triangle
              "-step", step,
              "-odir", file.path.system(type_os = params_general$type_os, dir_tile),
              "-otif",
              "-spike_free ", dist_freeze,
              ifelse(params_general$type_os == "Linux","2>&1", "") # crucial for Linux where LAStools seems to write to stderr
            ),
            timeout = timeout_lspikefree
          )
        }
        
        # recompute if spikefree aborted due to timeout
        # try out different combinations: sorting in different ways, or adding noise in increasing levels (20% of perturbation_max, 50% or 100%)
        perturbation_factors = c(0.0,0.0,0.2,0.5,1.0)
        j = 1
        while(return_system == 124 & j <= length(perturbation_factors)){
          
          # increasing the timeout 
          timeout_lspikefree = timeout_lspikefree * 2
          if(timeout_lspikefree > timeout_lspikefree_max) timeout_lspikefree = timeout_lspikefree_max
          if(timeout_lspikefree < timeout_lspikefree_min) timeout_lspikefree = timeout_lspikefree_min
          cat("New timeout:",timeout_lspikefree,"\n")
          
          # updating the perturbation factors
          perturbation_factor = perturbation_factors[j]
          perturbation_points = perturbation_max * perturbation_factor
          
          # we try two different ways of sorting the point cloud again before adding noise
          if(j <= 2){
            cat("Resorting point cloud\n")
            return_system = system(
              paste(
                file.path.system(type_os = params_general$type_os, params_general$path_lastools, update.command_lastools("lassort", type_architecture = params_general$type_architecture)),
                "-i", file.path.system(type_os = params_general$type_os, dir_tile, paste0(contours_pd_current$id,".laz")),
                "-cores", 1,
                ifelse(j == 1,paste0("-average ", 1000), ""), # this makes sure that the spatial sorting is adapted to the pulse/point densities (in this case, quadtree bucket size is chosen so as to cover an average of 1000 points)
                "-odir", file.path.system(type_os = params_general$type_os, dir_tile),
                "-olaz",
                "-odix _sort",
                ifelse(params_general$type_os == "Linux","2>&1", "") # crucial for Linux where LAStools seems to write to stderr
              )
            )
          } else {
            cat("Adding noise to point cloud:",perturbation_factor,"\n")
          }
          
          return_system = system(
            paste(
              file.path.system(type_os = params_general$type_os, params_general$path_lastools, update.command_lastools("las2dem", type_architecture = params_general$type_architecture)),
              "-i", file.path.system(type_os = params_general$type_os, dir_tile, paste0(contours_pd_current$id,"_sort.laz")), # new in v.42
              "-cores", 1,
              ifelse(params_general$bbtype == "tile","-use_tile_bb",""),
              # "-keep_first", # removed in v.42 and moved to splitting procedure
              ifelse(perturbation_points > 0,paste0("-translate_raw_xy_at_random ", round(perturbation_points/factor_scale,2)," ",round(perturbation_points/factor_scale,2)),""), # cf. https://groups.google.com/g/lastools/c/oY2gcW6w7HY/m/37zx06aJAQAJ
              "-kill", kill, # maximum size of triangle
              "-step", step,
              "-odir", file.path.system(type_os = params_general$type_os, dir_tile),
              "-otif",
              "-spike_free ", dist_freeze,
              ifelse(params_general$type_os == "Linux","2>&1", "") # crucial for Linux where LAStools seems to write to stderr
            ),
            timeout = timeout_lspikefree
          )
          
          # if successful with noise, then you write the result back to the topfolder
          if(return_system != 124 & perturbation_points > 0){
            contours_perturbation = copy(contours_pd_current)
            contours_perturbation$perturb_pt = perturbation_points
            file_contours = file.path(dir_processing,gsub(".laz",paste0("_pd",pd_current,"_perturbation.shp"),basename(tile_pointcloud)))
            writeVector(contours_perturbation, file_contours, overwrite = T)
          }
          
          j = j + 1 # update index
        }
        
        # read it in
        # check first whether spikefree algorithm succeeded, otherwise leave areas as NA
        # updated v.42 (use "sorted" output)
        if(file.size(file.path(dir_tile,paste0(contours_pd_current$id,"_sort.tif"))) > 0){
          dsm_spikefree_current = rast(file.path(dir_tile,paste0(contours_pd_current$id,"_sort.tif")))
          dsm_spikefree_current = mask(dsm_spikefree_current, contours_pd_current, touches = F)
          dsm_spikefree = ifel(is.na(dsm_spikefree), extend(dsm_spikefree_current,dsm_spikefree), dsm_spikefree)
        }
        
        time_end = Sys.time()
        diff_time_end = difftime(time_end,time_start,units = "secs")
        diff_time_endtotal = diff_time_endtotal + diff_time_end
        cat("Diff_time_end",diff_time_end,"\n")
        
        # resetting timeout
        timeout_lspikefree = as.integer(diff_time_end) * 25 # timeout is 25 times as long as actual algorithm needs
        if(timeout_lspikefree > timeout_lspikefree_max) timeout_lspikefree = timeout_lspikefree_max
        if(timeout_lspikefree < timeout_lspikefree_min) timeout_lspikefree = timeout_lspikefree_min
      }
      
      
      writeRaster(dsm_spikefree, filename = file_dsm_spikefree_tile, overwrite = T)
    }
    
    if(path_pdagg != ""){
      file_pd_agg_levels_tile = file.path(path_pdagg, gsub(".laz",".tif",basename(tile_pointcloud)))
      writeRaster(pd_agg_levels, filename =  file_pd_agg_levels_tile, overwrite = T)
    }
    # cleanup and return
    # cleanup.files(dir_tile)
    unlink(dir_tile, recursive = T)
    return(file_dsm_spikefree_tile)
  } else {
    return(NULL)
  }
}

make.dsm_pitfree_adaptive = function(tile_pointcloud, params_general, dir_processing, fact_agg = 5, buffer_contours = 10, step = 1, kill = 200, resolution_pitfree = c(0,5,10,20,30,40,50,75,100), subcircle = 0.1, factor_step = 1.0, limit_step = 0.2 * step, normalize = F, path_pdagg = ""){
  
  cat("Processing", basename(tile_pointcloud),"\n")
  # make folder to process in (best to separate)
  dir_tile = file.path(dir_processing,gsub(".laz","",basename(tile_pointcloud)))
  if(!dir.exists(dir_tile)) dir.create(dir_tile, recursive = T)
  
  # define dem algorithm
  lasgorithm_dem = ifelse(params_general$use.blast2dem == T, "blast2dem", "las2dem")
  
  # create pd raster and remove NA values inside the raster (to obtain correct pulse ensity)
  pd = rast(file.path(dir_pd, gsub(".laz",".tif",basename(tile_pointcloud))))
  names(pd) = "pd"
  
  mask_pd = as.polygons(clamp(pd,lower = 0, upper = 0, values = T), dissolve = T, values = T)
  mask_pd = fillHoles(mask_pd)
  mask_pd = rasterize(mask_pd,pd, "pd")
  
  pd_nona = sum(pd, mask_pd, na.rm = T)
  pd_agg = aggregate(pd_nona,fact = fact_agg, fun = "mean",na.rm = T)
  
  # now create separate pd levels
  pd_agg_levels = clamp(pd_agg, lower = 0.5, values = T)
  pd_agg_levels = ifel(pd_agg_levels < 3, floor(pd_agg_levels*2)/2, pd_agg_levels)
  pd_agg_levels = ifel(pd_agg_levels >= 3 & pd_agg_levels < 10, floor(pd_agg_levels), pd_agg_levels)
  pd_agg_levels = ifel(pd_agg_levels >= 10 & pd_agg_levels < 20, floor(pd_agg_levels/2) * 2, pd_agg_levels)
  pd_agg_levels = ifel(pd_agg_levels >= 20 & pd_agg_levels < 50, floor(pd_agg_levels/5) * 5, pd_agg_levels)
  pd_agg_levels = ifel(pd_agg_levels >= 50 & pd_agg_levels < 100, floor(pd_agg_levels/25) * 25, pd_agg_levels)
  
  # limit based on expected resolutions
  spacing_pulses_limit = limit_step/factor_step
  pd_limit = 1/{spacing_pulses_limit^2}
  
  if(pd_limit < 3){
    pd_limit = floor(pd_limit*2) / 2
  } else if(pd_limit >= 3 & pd_limit < 10){
    pd_limit = floor(pd_limit)
  } else if(pd_limit >= 10 & pd_limit < 20){
    pd_limit = floor(pd_limit/2) * 2
  } else if(pd_limit >= 20 & pd_limit < 50){
    pd_limit = floor(pd_limit/5) * 5
  } else if(pd_limit >= 50 & pd_limit < 100){
    pd_limit = floor(pd_limit/25) * 25
  } else {
    pd_limit = 100
  }
  pd_limit = max(pd_limit, 0.5)
  pd_agg_levels = clamp(pd_agg_levels, upper = pd_limit, values = T)
  
  # get pd clusters and process them one by one
  contours_pd = as.polygons(pd_agg_levels, values = T, trunc = F) # very important: trunc needs to be FALSE, otherwise integers are returned
  contours_pd$id = paste0(as.integer(contours_pd$pd),"dot",round((contours_pd$pd - as.integer(contours_pd$pd)) * 100)) # lastools does not seem to like actual dots in contour layers
  
  writeVector(contours_pd, filename = file.path(dir_tile, "contours_pd.shp"), overwrite = T)
  contours_pd = vect(file.path(dir_tile, "contours_pd.shp"))
  contours_pd_buffered = buffer_withsf(contours_pd, buffer_contours)
  contours_pd_buffered = fillHoles(contours_pd_buffered)
  
  # write to file
  path_contours_pd_buffered = file.path(dir_tile, "contours_pd_buffered.shp")
  writeVector(contours_pd_buffered, filename = path_contours_pd_buffered, overwrite = T)
  
  if(normalize == T){
    return_system = system(
      paste(
        file.path.system(type_os = params_general$type_os, params_general$path_lastools, update.command_lastools("lasheight", type_architecture = params_general$type_architecture)),
        "-i", file.path.system(type_os = params_general$type_os, tile_pointcloud),
        "-cores", 1,
        "-odir", file.path.system(type_os = params_general$type_os, dir_tile),
        "-olaz",
        "-replace_z",
        "-odix _norm",
        ifelse(params_general$type_os == "Linux","2>&1", "") # crucial for Linux where LAStools seems to write to stderr
      )
    )
    
    return_system = system(
      paste(
        file.path.system(type_os = params_general$type_os, params_general$path_lastools, update.command_lastools("lassort", type_architecture = params_general$type_architecture)),
        "-i", file.path.system(type_os = params_general$type_os, dir_tile, gsub(".laz","_norm.laz",basename(tile_pointcloud))),
        "-cores", 1,
        "-odir", file.path.system(type_os = params_general$type_os, dir_tile),
        "-olaz",
        "-odix _sort",
        ifelse(params_general$type_os == "Linux","2>&1", "") # crucial for Linux where LAStools seems to write to stderr
      )
    )
    
    return_system = system(
      paste(
        file.path.system(type_os = params_general$type_os, params_general$path_lastools, update.command_lastools("lasindex", type_architecture = params_general$type_architecture)),
        "-i",
        file.path.system(type_os = params_general$type_os, dir_tile, gsub(".laz","_norm_sort.laz",basename(tile_pointcloud))),
        "-cores", 1,
        ifelse(params_general$type_os == "Linux","2>&1", "") # crucial for Linux where LAStools seems to write to stderr
      )
    )
    
    return_system = system(
      paste(
        file.path.system(type_os = params_general$type_os, params_general$path_lastools, update.command_lastools("lasclip", type_architecture = params_general$type_architecture)),
        "-i", file.path.system(type_os = params_general$type_os, dir_tile, gsub(".laz","_norm_sort.laz",basename(tile_pointcloud))),
        "-cores", 1,
        "-odir", file.path.system(type_os = params_general$type_os, dir_tile),
        "-olaz",
        "-poly", file.path.system(type_os = params_general$type_os, path_contours_pd_buffered),
        "-split id",
        ifelse(params_general$type_os == "Linux","2>&1", "") # crucial for Linux where LAStools seems to write to stderr
      )
    )
  } else {
    return_system = system(
      paste(
        file.path.system(type_os = params_general$type_os, params_general$path_lastools, update.command_lastools("lasclip", type_architecture = params_general$type_architecture)),
        "-i", file.path.system(type_os = params_general$type_os, tile_pointcloud),
        "-cores", 1,
        "-odir", file.path.system(type_os = params_general$type_os, dir_tile),
        "-olaz",
        "-poly", file.path.system(type_os = params_general$type_os, path_contours_pd_buffered),
        "-split id",
        ifelse(params_general$type_os == "Linux","2>&1", "") # crucial for Linux where LAStools seems to write to stderr
      )
    )
  }
  
  # create an empty raster with same resolution as pd, but NA values
  dsm_pitfree = pd
  values(dsm_pitfree) = NA
  diff_time_endtotal = 0
  # i = 1
  # i = i + 1
  
  for(i in 1:nrow(contours_pd)){
    cat("Creating pitfree DSM for level",i,"\n")
    
    # first determine pd level and select the current polygons
    contours_pd_current = contours_pd[i]
    pd_current = contours_pd_current$pd
    id_current = contours_pd_current$id
    
    spacing_pulses_current = sqrt(1/pd_current)
    step_current = round(factor_step * spacing_pulses_current,2)
    thinning = 0.5 * step_current
    
    time_start = Sys.time()
    
    # first we thin the data
    return_system = system(
      paste(
        file.path.system(type_os = params_general$type_os, params_general$path_lastools, update.command_lastools("lasthin", type_architecture = params_general$type_architecture)),
        "-i",
        file.path.system(type_os = params_general$type_os, dir_tile,paste0(contours_pd_current$id,".laz")),
        "-cores", 1,
        "-highest",
        "-subcircle",subcircle,
        "-step", thinning, # usually half-step!
        "-odir", file.path.system(type_os = params_general$type_os, dir_tile),
        "-odix", "_circlethinned",
        "-olaz",
        ifelse(params_general$type_os == "Linux","2>&1", "") # crucial for Linux where LAStools seems to write to stderr
      )
    )
    
    # create a simple dsm, based on highest returns
    return_system = system(
      paste(
        file.path.system(type_os = params_general$type_os, params_general$path_lastools, update.command_lastools("lasgrid", type_architecture = params_general$type_architecture)),
        "-i",
        file.path.system(type_os = params_general$type_os, dir_tile,paste0(contours_pd_current$id,"_circlethinned.laz")),
        "-cores", 1,
        "-step", step,
        "-highest",
        ifelse(params_general$bbtype == "tile","-use_tile_bb",""),
        "-o", file.path.system(type_os = params_general$type_os, dir_tile,paste0(contours_pd_current$id,"_highest.tif")),
        ifelse(params_general$type_os == "Linux","2>&1", "") # crucial for Linux where LAStools seems to write to stderr
      )
    )
    
    # create a temporary dtm
    return_system = system(
      paste(
        file.path.system(type_os = params_general$type_os, params_general$path_lastools, update.command_lastools(lasgorithm_dem, type_architecture = params_general$type_architecture)),
        "-i", file.path.system(type_os = params_general$type_os, dir_tile,paste0(contours_pd_current$id,"_circlethinned.laz")),
        "-cores", 1,
        "-step", step_current,
        "-drop_z_above",0.1,
        ifelse(params_general$bbtype == "tile","-use_tile_bb",""),
        "-kill", kill,
        "-o", file.path.system(type_os = params_general$type_os, dir_tile,paste0(contours_pd_current$id,"_ground.tif")),
        ifelse(params_general$type_os == "Linux","2>&1", "") # crucial for Linux where LAStools seems to write to stderr
      )
    )
    
    file_highest = rast(file.path(dir_tile,paste0(contours_pd_current$id,"_highest.tif")))
    min_elevation = floor(as.numeric(global(file_highest,"min",na.rm = T)))
    min_elevation = max(min_elevation, 0) # remove below zero
    max_elevation = ceiling(as.numeric(global(file_highest,"max",na.rm = T)))
    
    # determine layers to go through for pitfree algorithm
    if(length(resolution_pitfree) == 1){
      resolution_pitfree = seq(min_elevation,max_elevation-resolution_pitfree,resolution_pitfree)
    }
    
    layers_pitfree = resolution_pitfree[resolution_pitfree >= min_elevation & resolution_pitfree < max_elevation]
    
    files_temp = file.path(dir_tile,paste0(contours_pd_current$id,"_ground.tif"))
    for(z in layers_pitfree){
      return_system = system(
        paste(
          file.path.system(type_os = params_general$type_os, params_general$path_lastools, update.command_lastools(lasgorithm_dem, type_architecture = params_general$type_architecture)),
          "-i", file.path.system(type_os = params_general$type_os, dir_tile,paste0(contours_pd_current$id,"_circlethinned.laz")),
          "-cores", 1,
          "-kill", 3.0 * step_current,
          "-step", step_current,
          "-drop_z_below", z,
          ifelse(params_general$bbtype == "tile","-use_tile_bb",""),
          "-o", file.path.system(type_os = params_general$type_os, dir_tile,paste0(contours_pd_current$id,"_temp_",z,".tif")),
          ifelse(params_general$type_os == "Linux","2>&1", "") # crucial for Linux where LAStools seems to write to stderr
        )
      )
      files_temp = c(files_temp, file.path(dir_tile,paste0(contours_pd_current$id,"_temp_",z,".tif")))
    }
    
    # now merge them all together
    files_exist = unlist(lapply(files_temp, file.exists))
    files_temp = files_temp[files_exist]
    files_temp_sizes = unlist(lapply(files_temp, file.size))
    files_temp = files_temp[files_temp_sizes > 0]
    
    # if any files were produced, merge them
    if(length(files_temp) > 0){
      dsm_temp = rast(files_temp)
      dsm_pitfree_current = max(dsm_temp, na.rm = T)
      # now resample
      dsm_pitfree_current = resample(dsm_pitfree_current, dsm_pitfree)
      dsm_pitfree_current = mask(dsm_pitfree_current, contours_pd_current, touches = F)
      dsm_pitfree = ifel(is.na(dsm_pitfree), dsm_pitfree_current, dsm_pitfree)
      
      # jpeg(file = file.path("/Users/vh21364/Documents",paste0(contours_pd_current$id,"_partial.jpeg")),width = 800,height = 800)
      # plot(dsm_pitfree, col = viridis(100), range = c(0,55), main = paste0(contours_pd_current$id, " pulses per m2"))
      # dev.off()
    }
    
    time_end = Sys.time()
    diff_time_end = difftime(time_end,time_start,units = "secs")
    diff_time_endtotal = diff_time_endtotal + diff_time_end
    cat("Diff_time_end",diff_time_end,"\n")
  }
  
  # now we write it out to the original directory
  file_dsm_pitfree_tile = file.path(dir_processing,gsub(".laz",".tif",basename(tile_pointcloud)))
  
  writeRaster(dsm_pitfree, filename = file_dsm_pitfree_tile, overwrite = T)
  
  if(path_pdagg != ""){
    file_pd_agg_levels_tile = file.path(dir_processing, "pdagg", gsub(".laz",".tif",basename(tile_pointcloud)))
    writeRaster(pd_agg_levels, filename =  file_pd_agg_levels_tile, overwrite = T)
  }
  
  # cleanup and return
  cleanup.files(dir_tile)
  return(file_dsm_pitfree_tile)
}

make.dsm_locallyadaptive = function(params_general = params_general, path_output = "", name_raster = "", kill = 200, step = 1, option = "spikefree", params_dsmadaptive = data.table(multi = 3.1, slope = 1.75, offset = 2.1), normalize = F, perturbation_max = 0.1, timeout_lspikefree_max = 600){
  
  
  # # params_general = params_general
  # path_output = file.path(params_general$path_data, name_dsm_lspikefree)
  # name_raster = name_dsm_lspikefree
  # kill = 200
  # step = resolution
  # option = "spikefree"
  # params_dsmadaptive = params_dsmadaptive_current
  # normalize = F
  
  # name_raster = "dsm_lspikefree"
  # path_data = "/home/karl/work/data/tmp/GCA_open_Luki/las2las/lasduplicate/lasnoise/lasground_new/height_lim125/"
  # path_output = file.path(path_data, name_raster)
  # params_dsmadaptive = data.table(multi = 3.1, slope = 1.75, offset = 2.1)
  # step = 1
  # n_cores = 15
  
  # record time
  time_start = Sys.time()
  
  # algorithm
  algorithm_locallyadaptive = paste0("make.dsm_", option,"_adaptive")
  
  # define output directory
  if(path_output == ""){
    if(name_raster == ""){
      path_output = file.path(params_general$path_data, paste0("dsm_l", option, "_multi",params_dsmadaptive$multi,"_slope",params_dsmadaptive$slope,"_offset",params_dsmadaptive$offset))
    } else {
      path_output = file.path(params_general$path_data, name_raster)
    }
  }
  if(!dir.exists(path_output)) dir.create(path_output)
  
  cat("Open source lspikefree")
  
  print(path_output)
  print(params_dsmadaptive)
  
  ctg <- readLAScatalog(params_general$path_data)
  
  # ctg <- readLAScatalog(path_data)
  crs_scan <- crs(ctg)
  
  opt_filter(ctg) = "-keep_first"
  opt_chunk_size(ctg)   <- size_tile
  opt_chunk_buffer(ctg) <- size_buffer
  opt_output_files(ctg) <- file.path(path_output, "chunk_{ID}_lspikefree")
  opt_progress(ctg) <- T
  
  # future::plan(future::multisession, workers = params_general$n_cores)
  future::plan(future::multisession, workers = params_general$n_cores)
  
  
  r <- rasterize_canopy(ctg, res = step, lspikefree(height_buffer = 0.5))
  future::plan(future::sequential)
  gc()
  
  # # create a subdirectory for the pulse density aggregation
  # path_pdagg = file.path(path_output, "pdagg")
  # if(!dir.exists(path_pdagg)) dir.create(path_pdagg)
  # 
  # # process in parallel
  # nbcluster = params_general$n_cores
  # cl = makeCluster(nbcluster)
  # clusterExport(cl, varlist = list(algorithm_locallyadaptive,"file.path.system","cleanup.files","call.lastools","buffer_withsf","update.command_lastools")) # export function, v.42 added buffer_withsf
  # clusterEvalQ(
  #   cl,
  #   {
  #     library(data.table) # for %like% operator etc.
  #     library(terra)
  #     library(parallel)
  #     library(sf) # required for buffer_withsf
  #     library(lidR) # required for readLASheader (v.42)
  #   }
  # )
  # on.exit(stopCluster(cl))  # on.exit is also triggered when an error occurs
  # 
  # # we use load balancing (parLapplyLB): improves processing time when there are asymmetries in tile size (e.g. edge vs. center)
  # # to avoid single large tiles blocking processing towards the end, we sort by tile size
  # tiles_pointcloud = data.table(file_tile = list.files(params_general$path_data, pattern = ".laz", full.names = T))
  # tiles_pointcloud[, size_file := file.size(file_tile)]
  # setorder(tiles_pointcloud, -size_file) # sort in descending order so that smallest files are dealt with last
  # tiles_pointcloud = tiles_pointcloud$file_tile
  # 
  # if(option == "spikefree"){
  #   
  #   # debugging help
  #   # files_dsm = as.character()
  #   # for(j in 1:length(tiles_pointcloud)){
  #   #   cat(j,"\n")
  #   #   files_dsm_current = make.dsm_spikefree_adaptive(tile_pointcloud, params_general = params_general, dir_processing = path_output, fact_agg = 5, buffer_contours = 10, step = step, kill = kill, multi_freeze = params_dsmadaptive$multi, slope_freeze = params_dsmadaptive$slope, offset_freeze = params_dsmadaptive$offset, path_pdagg = path_pdagg, perturbation_max = 0.05)
  #   #   files_dsm = c(files_dsm,files_dsm_current)
  #   # }
  #   
  #   files_dsm = parLapplyLB(cl, tiles_pointcloud, algorithm_locallyadaptive, params_general = params_general, dir_processing = path_output, fact_agg = 5, buffer_contours = 10, step = step, kill = kill, multi_freeze = params_dsmadaptive$multi, slope_freeze = params_dsmadaptive$slope, offset_freeze = params_dsmadaptive$offset, path_pdagg = path_pdagg, perturbation_max = perturbation_max, timeout_lspikefree_max = timeout_lspikefree_max, step_thinning = 0.1)
  # }
  # 
  # # cleanup
  # tmpFiles(old = T, remove = T)
  # tmpFiles(orphan = T, remove = T)
  
  # calculate elapsed time
  time_end = Sys.time()
  time_processing = difftime(time_end,time_start,units = "mins")
  return(time_processing)
}

compute.sumstats_pc = function(params_general, path_output = "", resolution = 100, cutoff_height = 0.0, type_output = "tif"){
  # define output directory
  if(path_output == ""){
    path_output = file.path(params_general$path_data,"sumstats_pc")
  }
  if(!dir.exists(path_output)) dir.create(path_output)
  
  if(!is.voidstring(params_general$path_lastools)){
    # first we compute the metrics from the point cloud
    call.lastools(command_lastools = "lascanopy"
                  , arguments_lastools = paste(
                    "-drop_withheld"
                    , "-merged"
                    , "-p 5 25 50 75 90"
                    , "-min -max -avg -std"
                    , "-cov"
                    , "-dns"
                    , "-vci 1"
                    , "-height_limoff", cutoff_height
                    , "-step",resolution
                    , sep = " "
                  )
                  , path_output = path_output
                  , params_general =  params_general
                  , type_output = type_output
    )
  }
}

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
#### 4. High-level processing functions ####
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#

# this is the function that does all the heavy lifting
# applied to every data subset and processing the las files in it
# TODO: should be split into smaller chunks

process.datasubset = function(path_lastools, path_tmp, path_input, path_output, type_file, addendum_name = "", metadata = NULL, retile = T, deduplicate = T, denoise = T, reclassify = T, resolution = 1, n_cores = 4, size_tile = 500, size_buffer = 25, cleanup = T, nbclusters_forced = NULL, force.utm = F, remove.buffer = F, remove.vlr = F, remove.evlr = F, factor_rescale = NULL, path_output_lazclean = "", path_output_laznorm = "", types_dsm = c("tin","spikefree", "lspikefree"), params_dsmadaptive = data.table(multi = 3.1, slope = 1.75, offset = 2.1), resolution_sumstatspc = c(25,100), estimate.laserpenetration = F, type_os = "automatic", type_architecture = "64", perturbation_max = 0.1, timeout_lspikefree_max = 600, overwrite.crs = F, use.blast2dem = F, is.stdtime = NA, height_lim = 125, angle_lim = NULL, class_rm = c(), exclass_rm = c(), force.type_point = NULL, logfile = "", checkpoint_step){
  
  # remove temporary files (terra package)
  tmpFiles(old=TRUE, remove=TRUE)
  
  has.path_lastools = check.path_lastools(path_lastools = path_lastools)
  
  #if(has.path_lastools == T){
  if(T){
    # create a new temporary path and the output path
    if(!dir.exists(path_tmp)) dir.create(path_tmp, recursive = TRUE)
    if(!dir.exists(path_output)) dir.create(path_output, recursive = TRUE)
    
    # clean up the suffix / addendum_name (replacing white space)
    addendum_name = gsub("[[:space:]]", "_", addendum_name)
    addendum_name = gsub(":", "h", addendum_name,fixed = TRUE)
    if(addendum_name != ""){
      addendum_name = paste0("_",addendum_name)
    }
    
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
    cat("\nRemove old data from temporary directory and add new data\n")
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
    if(checkpoint_step == 0){copy.files.totmp(path_input = path_input, path_tmp = path_tmp, type_file = type_file)}
    
    # create a data.table that lists all las or laz files (their paths, names, and folders that they are in)
    folders_toprocess = data.table(file_toprocess = list.files(path_tmp, recursive = T, full.names = T, pattern = paste0(".",type_file)))
    folders_toprocess[, folder_toprocess := dirname(file_toprocess)]
    folders_toprocess[, name_file := basename(file_toprocess)] # does not seem to be needed at the moment
    
    # determine the unique folders
    uniquefolders_toprocess = unique(folders_toprocess$folder_toprocess)
    
    # now the actual processing can start
    params_general_baseline = data.table(
      path_lastools = path_lastools,
      path_data = uniquefolders_toprocess,
      type_file = type_file,
      bbtype = "",
      type_os = type_os,
      type_architecture = type_architecture,
      n_cores = n_cores,
      buffer = size_buffer,
      cleanup = cleanup,
      use.blast2dem = use.blast2dem
      # , logfile = logfile
    )
    
    # original output path
    path_output_baseline = path_output
    
    #%%%%%%%%%%%%%%%%%%%%%%%%#
    cat("\nRead in metadata\n")
    #%%%%%%%%%%%%%%%%%%%%%%%%#
    metadata_template = data.table(
      subfolder = character()
      , dataset = character()
      , site = character()
      , source = character()
      , citation = character()
      , contact = character()
      , access = character()
      , license = character()
      , updated = character()
      , id_orig = character()
      , acq_type = character()
      , acq_crs = character()
      , acq_units = character()
      , acq_mindt = character()
      , acq_maxdt = character()
      , acq_system = character()
      , acq_AGL = integer()
      , acq_swath = integer()
      , acq_wavel = integer()
      , acq_fpsize = numeric()
      , acq_lasdiv = numeric()
      , notes = character()
      , issues = character()
      , height_lim = numeric()
      , angle_lim = numeric()
      , class_rm = character()
      , exclass_rm = character()
    )
    
    if(is.null(metadata)) {
      # we look for the metadata sheet somewhere in the path
      file_metadata = character()
      
      # a little bit of path magic
      path_origin = path_input
      if(grepl(pattern = paste0(".",type_file),x = path_input, fixed = TRUE) & file_test("-f",path_input)){ # file test may be overkill here
        path_origin = dirname(path_input)
      }
      wd_processing = getwd();setwd(path_origin); path_absolute = getwd(); setwd(wd_processing)
      
      # now cycle through the absolute path
      length_filepath = length(tstrsplit(path_absolute,"/"))
      level_filepath = 0
      path_cluster = ""
      if(grepl(pattern = paste0(".",type_file),x = path_input, fixed = TRUE) & file_test("-f",path_input)){ # file test may be overkill here
        path_cluster = paste0(path_cluster,basename(path_input))
      }
      while(level_filepath < length_filepath & length(file_metadata) == 0){
        file_metadata = list.files(path_absolute, pattern = "_metadata.csv", full.names = T)
        level_filepath = level_filepath + 1
        path_cluster = file.path(basename(path_absolute),path_cluster)
        path_absolute = dirname(path_absolute)
      }
      
      if(length(file_metadata) == 1){
        metadata = fread(file_metadata)
        metadata[, subfolder := as.character(subfolder)]
        metadata[is.na(subfolder), subfolder := ""]
        metadata[, subfolder := trimws(subfolder)]
        metadata = metadata[mapply(grepl, subfolder, path_cluster, fixed = TRUE) | subfolder == "" | subfolder == "."]
        metadata[, subfolder_length := nchar(subfolder)]
        setorder(metadata, -subfolder_length)
        metadata = metadata[1]
        if(nrow(metadata) == 0) metadata = NULL
      }
    }
    
    if(is.null(metadata)) metadata = data.table(subfolder = as.character(NA))
    metadata = rbind(metadata_template, metadata, use.names = T, fill = T)
    
    # overwrite parameters with provided metadata 
    if(!is.null(metadata$height_lim) & !is.na(metadata$height_lim) & !is.nan(metadata$height_lim)){height_lim = min(height_lim, metadata$height_lim)}
    if(!is.null(metadata$angle_lim) & !is.na(metadata$angle_lim) & !is.nan(metadata$angle_lim)){angle_lim = min(angle_lim, metadata$angle_lim)}
    
    class_rm = paste(class_rm, collapse = " ")
    exclass_rm = paste(exclass_rm, collapse = " ")
    
    if(!is.voidstring(metadata$class_rm)){class_rm = metadata$class_rm}
    if(!is.voidstring(metadata$exclass_rm)){class_rm = metadata$exclass_rm}
    
    # new in v.47: options to force CRS depending on entry in metatadata
    # note that this is not yet 100% perfect: a feet CRS cannot yet be coerced and then converted to UTM
    if(like(metadata$acq_crs, "overwrite.with:", fixed = T)){
      overwrite.crs = T
      cat("WARNING! CRS of scan will be overwritten\n")
      if(file.exists(logfile)){
        logfile_dt = data.table(path = path_output_baseline, issue = paste0("WARNING! CRS of scan will be overwritten."))
        fwrite(logfile_dt, file = logfile, append = T)
      }
    }
    
    if(is.na(metadata$acq_mindt) | is.na(metadata$acq_maxdt)){
      cat("WARNING! No flight dates provided as part of metadata\n")
      if(file.exists(logfile)){
        logfile_dt = data.table(path = path_output_baseline, issue = paste0("WARNING! No flight dates provided as part of metadata."))
        fwrite(logfile_dt, file = logfile, append = T)
      }
    }
    
    force.utm_auto = F
    if(overwrite.crs == T){
      cat("CRS will be automatically overwritten from metadata. Attention: no reprojection of CRS possible at the moment\n")
    } else {
      if(force.utm == "from_metadata"){
        if(metadata$acq_units %like% "ft"){
          force.utm_auto = T 
        }
      } else {
        if(force.utm == T){
          force.utm_auto = T
        }
      }
    }
    
    # if (checkpoint_step <= 2){}
    
    if(retile != "include.adjacents"){
      #%%%%%%%%%%%%%%%%%%%%%%%#
      cat("\nClean up data\n")
      #%%%%%%%%%%%%%%%%%%%%%%%#
      # when using the "include.adjacents" option, this step will be made slightly later; otherwise workflow gets complicated; this option should remain restricted to highly standardized (e.g. government agency prepared) tiles anyways
      if(force.utm_auto == T){
        cat("\nConverting point cloud to UTM \n")
      }
      
      # OK for open source
      params_general_baseline = las2las.initial(params_general = params_general_baseline, metadata=metadata, size_tile=size_tile, force.utm = force.utm_auto,remove.vlr = F, remove.evlr = F, factor_rescale = factor_rescale, angle_lim = angle_lim, update.path = T, class_rm = class_rm, exclass_rm = exclass_rm, type_point = force.type_point)
      
      # TO DO for open source (secondary priority)
      lasinfo.repair(params_general = params_general_baseline)
    }
    
    # TO DO for open source (secondary priority)
    refine = NULL
    if(size_tile == "adaptive"){
      #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
      cat("\nPrepare adaptive tiling \n")
      #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
      lasinfo(params_general = params_general_baseline)
      load(file.path(params_general_baseline$path_data,"summary_full.RData"))
      density_points_initial = summary_full$density_points_mean
      
      refine = 1000000 # limit to 1 million
      limit_points_pertile = refine * 0.95
      extent_tile_adaptive = limit_points_pertile/density_points_initial
      size_tile_adaptive = sqrt(extent_tile_adaptive)
      size_tile = as.integer((size_tile_adaptive - 2 * size_buffer)/50) * 50
      if(size_tile > 500){
        size_tile = 500
      } else if(size_tile < 50){
        size_tile = 50
      }
      cat("\nSetting tile size to",size_tile,"\n")
    }
    
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
    cat("\nCheck for spatial clusters in data\n")
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
    
    # currently a bit slow for a single file, but it is consistent
    clusters_data = find.clusters_data(params_general = params_general_baseline, path_output = path_output_baseline, nbclusters_forced = nbclusters_forced)
    
    cat("\nFound", length(clusters_data),"clusters\n")
    outlines_input = vect(file.path(params_general_baseline$path_data, "outlines_input.shp"))
    
    nberrors = 0
    for(i in 1:length(clusters_data)){
      
      #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
      cat("\nPROCESSING CLUSTER",i,"\n")
      #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
      
      time_start_current = Sys.time()
      
      # we iteratively update params_general and create output folders
      params_general = clusters_data[[i]]
      path_output = path_output_baseline
      if(length(clusters_data) > 1) path_output = file.path(path_output,basename(params_general$path_data))
      if(!dir.exists(path_output)) dir.create(path_output, recursive = TRUE)
      
      step_processing = paste0("start cluster ",i)
      
      tryCatch({
        
        # compute a few statistics
        files_toprocess = list.files(params_general$path_data, pattern = paste0(".",params_general$type_file), full.names = T)
        n_files = length(files_toprocess)
        size_byte_input = sum(file.info(files_toprocess)$size)
        size_KB_input = round(size_byte_input/(1024),2)
        size_MB_input = round(size_byte_input/(1024*1024),2)
        
        # process
        if(size_KB_input >= 10){
          
          if(retile == "include.adjacents"){
            step_processing = paste0("retiling with adjacents")
            
            cat("\nFind adjacent las files\n")
            outlines_cluster = outlines_input[outlines_input$cluster == i]
            path_origin = path_input
            if(grepl(pattern = paste0(".",type_file),x = path_input, fixed = TRUE) & file_test("-f",path_input)){ # file test may be overkill here
              path_origin = dirname(path_input)
            }
            files_adjacent = add.adjacent(outlines_cluster = outlines_cluster, path_origin = path_input, path_moveto = params_general$path_data, type_file = params_general$type_file, buffer = params_general$buffer)
            cat("\nFound",length(files_adjacent),"adjacent files\n")
            
            cat("\nLasindex (COMMENTED)\n")
            # lasindex(params_general)
            
            if(remove.buffer == T){
              cat("Remove any existing buffers\n")
              params_general = call.lastools(command_lastools = "lastile",arguments_lastools = "-remove_buffer",params_general = params_general,update.path = T)
            }
            
            cat("\nRetiling\n")
            params_general = lastile(params_general = params_general,size_tile = size_tile,refine = refine, size_MB_input = size_MB_input,arguments_additional = paste0("-drop_withheld", paste0(" -drop_class 7 18 ", class_rm), ifelse(nchar(exclass_rm) > 0, paste0(" -drop_extended_class ", exclass_rm),""))) # new in v.47: remove extra noise classes or otherwise from scan (necessary for IGN France, for example, where artefacts/noise is usually marked with 65, or even 28)
            
            cat("\nRemove superfluous tiles\n")
            files_removed = remove.adjacent(path_tiles = params_general$path_data, outlines_cluster = outlines_cluster, type_file = params_general$type_file, size_tile = size_tile)
            cat("\nRemoved",length(files_removed),"tiles\n")
            
            arguments_additional = paste0(ifelse(!is.null(angle_lim),paste0("-drop_abs_scan_angle_above ", angle_lim," "),""), ifelse(force.utm_auto == T,"-target_utm auto ",""),ifelse(!is.null(factor_rescale),paste0(" -rescale ",factor_rescale," ",factor_rescale," ",factor_rescale," "),""))
            if(arguments_additional != ""){
              cat("\nAdditional cleanup\n")
              # usually performed beforehand
              params_general = call.lastools(command_lastools = "las2las",arguments_lastools = arguments_additional,params_general = params_general,update.path = T)
            }
            
          } 
          else if(retile == T){
            # TODO
            # step_processing = paste0("retiling")
            # cat("\nLasindex\n")
            # lasindex(params_general)
            # 
            # if(remove.buffer == T){
            #   cat("Remove any existing buffers\n")
            #   params_general = call.lastools(command_lastools = "lastile",arguments_lastools = "-remove_buffer",params_general = params_general,update.path = T)
            # }
            # 
            # cat("\nRetiling\n")
            # params_general = lastile(params_general = params_general,size_tile = size_tile, refine = refine, size_MB_input = size_MB_input)
          }
          
          #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
          cat("\nGenerating output from untreated point cloud\n")
          #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
          
          step_processing = paste0("initial classification")
          
          params_general_nocleanup = copy(params_general)
          params_general_nocleanup$cleanup = F
          
          cat("\nGenerate initial classifications raster called...\n")
          
          if(!is.voidstring(params_general$path_lastools))
          {
            call.lastools(command_lastools = "lasgrid", 
                          arguments_lastools = paste0("-classification -step ",resolution), 
                          type_output = "tif", 
                          params_general = params_general_nocleanup,
                          update.path = F)
          }
          else
          {
            cat("\nOpen source version\n")
            
            path_output_local = file.path(params_general$path_data, "lasgrid")
            if(!dir.exists(path_output_local)) dir.create(path_output_local, recursive = TRUE)
            
            # files.input = list.files(path = params_general$path_data, pattern = "\\.laz", full.names = TRUE)
            # files_names.input = list.files(path = params_general$path_data, pattern = "\\.laz", full.names = FALSE)
            
            ctg <- readLAScatalog(params_general$path_data)
            crs_scan <- crs(ctg)
            
            opt_chunk_size(ctg)   <- size_tile
            opt_chunk_buffer(ctg) <- size_buffer
            opt_output_files(ctg) <- file.path(path_output_local, "chunk_{ID}_initial_classif")
            opt_progress(ctg) <- T
            
            future::plan(future::multisession, workers = params_general$n_cores)
            
            majority_class <- function(z) {
              v = which.max(table(z))
              v = as.numeric(names(v))
              return(v)
            }
            
            catalog_apply(ctg, function(las_chunk) {
              r <- grid_metrics(las_chunk, ~majority_class(Classification), res = resolution)
              return(r)}
            )
            
            future::plan(future::sequential)
            gc()
            
          }
          
          cat("\nClassification files analysis called...\n")
          files_classification = list.files.nonzero(path = file.path(params_general$path_data, "lasgrid"), pattern = ".tif", full.names = TRUE)
          classifications = terra::vrt(files_classification)
          crs_scan = terra::crs(classifications)
          
          # update crs based on path, if no crs is available from lidar file
          # assign the same crs to all rasters
          if(is.na(crs_scan) | is.null(crs_scan) | crs_scan == "" | overwrite.crs == T){
            crs_scan = get.crs_from_path(path_file = path_input)
            
            if(is.na(crs_scan) | is.null(crs_scan) | crs_scan == ""){
              crs_scan = gsub("overwrite.with:", "", metadata$acq_crs, fixed = T) # new in v.47: allow an overwrite.with option
              terra::crs(classifications) = crs_scan
            } else {
              # TODO
              terra::crs(classifications) = crs_scan
            }
          }
          
          classifications_unique = unique(classifications,na.rm = T)[,1]
          cat("\nClassification files analysis done!\n")
          
          cat("\nGenerate classifications raster called...\n")
          if(length(classifications_unique) > 1){
            # only create output when it is informative
            writeRaster(classifications, filename = file.path(path_output,paste0("classifications_supplied",addendum_name,".tif")), overwrite = T)
          }
          if(params_general$cleanup == T) file.remove(files_classification)
          cat("\nGenerate classifications raster done!\n")
          
          # if(!is.voidstring(params_general$path_lastools))
          # {
          #   # now we can safely delete user information
          #   if(remove.vlr == T & remove.evlr == F){
          #     params_general = call.lastools(command_lastools = "las2las",arguments_lastools = "-remove_all_vlrs",type_output = "laz",update.path = T,params_general = params_general)
          #   } else if(remove.vlr == F & remove.evlr == T){
          #     params_general = call.lastools(command_lastools = "las2las",arguments_lastools = "-remove_all_evlrs",type_output = "laz",update.path = T,params_general = params_general)
          #   } else if(remove.vlr == T & remove.evlr == T){
          #     params_general = call.lastools(command_lastools = "las2las",arguments_lastools = "-remove_all_vlrs -remove_all_evlrs",type_output = "laz",update.path = T,params_general = params_general)
          #   }
          # }
          
          # TODO: mettre le code de JR dans une fonction remove overhanging points
          
          cat("Open source REMOVE OVERHANGING POINTS")
          
          # # Remove overhangings points
          # remove_hoverhangs_ground_points = function(res = 1)
          # {
          #   gnd = filter_ground(las)
          #   gnd = lidR:::subcircle(gnd, r = res*1.05, n = 16)
          #   dtm_max = rasterize_canopy(gnd, res)
          #   dtm_max[is.na(dtm_max)] = -Inf
          #   las = merge_spatial(las, dtm_max, "tmp")
          #   rm = las$Z < las$tmp - 10 & las$Classification == lidR::LASGROUND
          #   las@data$tmp = NULL
          #   las = las[!rm]
          # }
          # 
          # ctg_path = params_general$path_data
          # ctg = lidR::readALScatalog(ctg_path)
          # 
          # catalog_apply(ctg, remove_hoverhangs_ground_points())
          
          
          cat("\nGenerate dtm_nooverhangs called...\n")
          
          if(any(classifications_unique == 2)){
            # only create a DTM when there are ground points in the supplied classification
            # (presupposes that at least one ground pixel is visible, but that should be a given)
            status = tryCatch(
              {
                make.dtm_nooverhangs(params_general = params_general, path_output = "", path_products = path_output, name_raster = "dtm_supplied", kill = 200, step = resolution, type_output = "tif", threshold_drop = 10, classes_ground = "2 8",arguments_additional = "")
                # las2dem(params_general = params_general, path_output = "", name_raster = "dtm_supplied", kill = 200, step = resolution, type_output = "tif", option = "dtm")
                #
              },
              error = function(e){
                return("DTM could not be obtained\n")
              }
            )
          }
          cat("\nGenerate dtm_nooverhangs done!\n")
          
          #%%%%%%%%%%%%%%%%%%%%%%%%%%%#
          cat("\nCheck reprojection\n")
          #%%%%%%%%%%%%%%%%%%%%%%%%%%%#
          
          step_processing = paste0("reprojection check")
          
          # new in v.47: we check this early on so that processing fails already at this step if something is wrong with CRS
          if(is.na(crs_scan)){
            cat("ERROR! No coordinate reference system detected either in metadata or in point cloud. The pipeline will fail. Provide correctly referenced metadata or update the point cloud.\n")
            if(file.exists(logfile)){
              logfile_dt = data.table(path = path_output, issue = paste0("ERROR! No coordinate reference system detected either in metadata or in point cloud. The pipeline will fail. Provide correctly referenced metadata or update the point cloud."))
              fwrite(logfile_dt, file = logfile, append = T)
            }
          }
          
          extent_scan = as.polygons(ext(classifications))
          terra::crs(extent_scan) = crs_scan
          #extent_scan_projected = project(extent_scan,"EPSG:4326")
          
          #%%%%%%%%%%%%%%%%%%%%%%%%%%#
          cat("\nCustom processing\n")
          #%%%%%%%%%%%%%%%%%%%%%%%%%%#
          
          if(deduplicate == T){
            step_processing = paste0("deduplicate")
            cat("\nDuplicate (xyz) removal\n")
            params_general = lasduplicate(params_general)
          }
          
          if(!is.voidstring(params_general$path_lastools))
          {
            # Sorting to be implemented in open source
            params_general = call.lastools(command_lastools = "lassort",arguments_lastools = "", params_general = params_general)
            
            #%%%%%%%%%%%%%%%%%%%%%%%%%%#
            cat("\nGet summary stats\n")
            #%%%%%%%%%%%%%%%%%%%%%%%%%%#
            
            # Info to be implemented in open source
            lasinfo(params_general = params_general)
            load(file.path(params_general$path_data,"summary_bytile.RData"))
            load(file.path(params_general$path_data,"summary_full.RData"))
            
            # noise removal
            # 'step' gives the cube size (e.g., 4m), 'isolated' the number of points
            # slightly modified since v.43: more conservative/less aggressive, as this now takes the 10th percentile of point densities across tiles and multiplies by 0.5 instead of the mean / step has been increased to 4m
            # reasoning: the most critical part is not to accidentally remove ground points. Keeping a few birds or cloud points seems less of an issue in comparison, as they will be far more localized and easy to spot
            # should have a small impact on scans with little pulse density variation or low density scans, where the minimum of 5 points is usually significant
            # TODO: this function should actually be made a locally adaptive one
            
            # Denoise to be implemented in open source
            if(denoise == T){
              step_processing = paste0("denoise")
              cat("\nNoise removal\n")
              step_noise = 4
              density_points_cutoff = 0.5 * quantile(summary_bytile$density_points,0.1)
              isolated = as.integer(0.007 * density_points_cutoff * (3 * step_noise)^2)
              if(isolated < 5) isolated = 5
              params_general = lasnoise(params_general = params_general, step = step_noise, isolated = isolated)
            }
            
            # Buffer removal to be implemented in open source
            if(path_output_lazclean != ""){
              step_processing = paste0("write laz files")
              if(!dir.exists(path_output_lazclean)) dir.create(path_output_lazclean)
              
              # before outputting, we remove the buffers
              call.lastools(command_lastools = "lastile",arguments_lastools = "-remove_buffer",params_general = params_general,path_output = path_output_lazclean, type_output = "laz", update.path = F, return.time = F)
            }
          }
          else{
            
            # Info to be implemented in open source
            lasinfo(params_general = params_general)
            load(file.path(params_general$path_data,"summary_bytile.RData"))
            load(file.path(params_general$path_data,"summary_full.RData"))
            
            #Noise removal
            cat("Open source denoising ! ")
            path_output_denoising = file.path(params_general$path_data, "lasnoise")
            if(!dir.exists(path_output_denoising)) dir.create(path_output_denoising)
            output_filename <- file.path(path_output_denoising, "*.laz")
            
            # files <- list.files(file.path.system(type_os = params_general$type_os, params_general$path_data), 
            #                     pattern = "*.laz$", full.names = TRUE)
            
            # 1. Remova o uso da função 'file.path.system' aqui (deixe o caminho limpo para o R)
            path_r <- file.path(params_general$path_data)
            
            # 2. Use o padrão corrigido com a busca sem diferenciar maiúsculas/minúsculas
            files <- list.files(
              path_r, 
              pattern = "\\.laz$", 
              ignore.case = TRUE, 
              full.names = TRUE
            )
            
            # 3. Quando for passar o 'files' para o LAStools/prompt de comando (via system), 
            # aí sim você adiciona as aspas para o Windows não se perder com espaços:
            if (params_general$type_os == "Windows") {
              files_lastools <- paste0('"', files, '"')
            } else {
              files_lastools <- files
            }
            
            step_noise = 4
            density_points_cutoff = 0.5 * quantile(summary_bytile$density_points, 0.1)
            isolated = as.integer(0.007 * density_points_cutoff * (3 * step_noise)^2)
            if(isolated < 5) isolated = 5
            
            pipeline = lasR::classify_with_ivf(res = step_noise, n = isolated, class = 18L) + 
              lasR::write_las(output_filename)
            
            ans = exec(pipeline, 
                       on = files, 
                       ncores = params_general$n_cores,  
                       progress = TRUE)
          }
          
          time_ground_basic = as.numeric(NA)
          time_ground_refinement = as.numeric(NA)
          
          if(params_general$cleanup == T) cleanup.files(params_general$path_data)
          
          cat("Amending file path and reindexing\n")
          params_general$path_data = path_output_denoising
          params_general$type_file = "laz"
          # lasindex(params_general)
          cat("Amended file path returned\n")
          
          if(reclassify == T){
            
            step_processing = paste0("reclassify")
            cat("\nGround classification\n")
            start_time = Sys.time()
            params_general = lasground_new(params_general = params_general, path_output = "", update.path = TRUE, arguments_additional = "")
            end_time = Sys.time()
            time_ground_basic = as.numeric(difftime(end_time,start_time,units = "mins"))
            
            if(!is.voidstring(params_general$path_lastools))
            {
              # new in v.48: refinement of ground classification
              cat("\nRefinement of ground classification\n")
              step_processing = paste0("refine ground classification")
              start_time = Sys.time()
              params_general = refine.ground(params_general = params_general, path_output = "",update.path = T)
              end_time = Sys.time()
              time_ground_refinement = as.numeric(difftime(end_time,start_time,units = "mins"))
            }
          }
          
          if(height_lim > 0){
            step_processing = paste0("drop heights above ", height_lim)
            cat("\nDrop canopy heights above",height_lim,"\n")
            params_general = filter.height(params_general = params_general, path_output = "", height_lim = height_lim, update.path = TRUE)
          }
          
          #%%%%%%%%%%%%%%%%%%%%%%%%%%#
          cat("\nGet summary stats\n")
          #%%%%%%%%%%%%%%%%%%%%%%%%%%#
          
          lasinfo(params_general = params_general)
          load(file.path(params_general$path_data,"summary_bytile.RData"))
          load(file.path(params_general$path_data,"summary_full.RData"))
          
          # add infos on ground classification
          summary_full$time_ground_basic = time_ground_basic
          summary_full$time_ground_refinement = time_ground_refinement
          
          #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
          cat("\nGet pulse density and scan angle\n")
          #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
          
          scan_angle_field = get.scan_angle_field(
            path_data  = params_general$path_data, 
            type_file  = params_general$type_file
          )
          
          step_processing = paste0("get pulsedensity rasters")
          get.pulsedensity(params_general = params_general, path_output = "", step = resolution, scan_angle_field = scan_angle_field)
          
          files_pulsedensity_firstlast = list.files.nonzero(path = file.path(params_general$path_data,"pulsedensity"),
                                                            pattern = "pulsedensity_[0-9]+\\.tif$", full.names = TRUE)
          # using the virtual raster method from terra package
          pulsedensity_firstlast = vrt(files_pulsedensity_firstlast)
          names(pulsedensity_firstlast) = c("first","last")
          terra::crs(pulsedensity_firstlast) = crs_scan
          writeRaster(pulsedensity_firstlast, filename = file.path(path_output,paste0("pulsedensity_firstlast",addendum_name,".tif")), overwrite = T)
          
          files_pulsedensity_scanangle = list.files.nonzero(path = file.path(params_general$path_data,"pulsedensity"),
                                                            pattern = "pulsedensity_scan_angle[0-9]+\\.tif$", full.names = TRUE)
          # using the virtual raster method from terra package
          pulsedensity_scanangle = vrt(files_pulsedensity_scanangle)
          terra::crs(pulsedensity_scanangle) = crs_scan
          writeRaster(pulsedensity_scanangle, filename = file.path(path_output,paste0("pulsedensity_scanangle",addendum_name,".tif")), overwrite = T)
          
          # create a pulse density mask (smooth within a radius of 5m)
          # use median smoothing, so that a single pixel with high pulse density does not unduly influence the smoothed pulse
          pulsedensity_first = terra::subset(pulsedensity_firstlast, "first")
          pulsedensity_nona = ifel(!is.na(pulsedensity_first), pulsedensity_first, 0)
          pulsedensity_smoothed = terra::focal(pulsedensity_nona, w = 11, fun = "median", na.rm = T)
          mask_pd02 = clamp(pulsedensity_smoothed, lower = 2, values = F)
          mask_pd02 = clamp(mask_pd02, upper = 1, values = T)
          mask_pd04 = clamp(pulsedensity_smoothed, lower = 4, values = F)
          mask_pd04 = clamp(mask_pd04, upper = 1, values = T)
          
          writeRaster(mask_pd02, filename = file.path(path_output,paste0("mask_pd02",addendum_name,".tif")), overwrite = T)
          writeRaster(mask_pd04, filename = file.path(path_output,paste0("mask_pd04",addendum_name,".tif")), overwrite = T)
          
          # step_processing = paste0("get pulsedensity_scanangle rasters")
          
          # # pulse density rasters, filtering for scan angles
          # for(scanangle_abs_current in c(20)){
          #   get.pulsedensity(params_general = params_general,path_output = file.path(params_general$path_data,"pulsedensity",paste0("pulsedensity",scanangle_abs_current)), step = resolution, scanangle_abs_max = scanangle_abs_current)
          #
          #   files_pulsedensity_scanangle = list.files.nonzero(path = file.path(params_general$path_data,"pulsedensity",paste0("pulsedensity",scanangle_abs_current)), pattern = ".tif", full.names = TRUE)
          #   # using the virtual raster method from terra package
          #   pulsedensity_scanangle = vrt(files_pulsedensity_scanangle)
          #   terra::crs(pulsedensity_scanangle) = crs_scan
          #   writeRaster(pulsedensity_scanangle, filename = file.path(path_output,paste0("pulsedensity_scanangle",scanangle_abs_current,addendum_name,".tif")), overwrite = T)
          #   unlink(x = file.path(path_output,paste0("pulsedensity",scanangle_abs_current)), recursive = T) # not needed for further processing
          #
          # }
          
          step_processing = paste0("get pulsedensity_lastreturn rasters")
          # pulse density, based on last returns
          # get.pulsedensity(params_general = params_general,path_output = file.path(params_general$path_data,"pulsedensity","pulsedensity_lastreturn"), step = resolution, keep_first = F)
          # files_pulsedensity_lastreturn = list.files.nonzero(path = file.path(params_general$path_data,"pulsedensity","pulsedensity_lastreturn"), pattern = ".tif", full.names = TRUE)
          # using the virtual raster method from terra package
          pulsedensity_last = terra::subset(pulsedensity_firstlast, "last")
          # pulsedensity_lastreturn = vrt(pulsedensity_last)
          terra::crs(pulsedensity_last) = crs_scan
          writeRaster(pulsedensity_last, filename = file.path(path_output,paste0("pulsedensity_lastreturn",addendum_name,".tif")), overwrite = T)
          unlink(x = file.path(path_output,"pulsedensity_lastreturn"), recursive = T) # not needed for further processing
          
          step_processing = paste0("get scanangle_abs rasters")
          # same for scan angle
          # special tryCatch condition, in case there are problems with the scan angle field
          angle99th = NULL
          status = tryCatch(
            {
              get.scanangle_abs(params_general = params_general,path_output = "", step = resolution, scan_angle_field = scan_angle_field)
              
              files_scanangle = list.files.nonzero(path = file.path(params_general$path_data,"scanangle_abs"), pattern = "\\.tif", full.names = TRUE)
              
              scanangle = vrt(files_scanangle);
              terra::crs(scanangle) = crs_scan
              #scanangle = ifel(is.na(scanangle) & !is.na(pulsedensity), 0, scanangle) # scanangles below a certain threshold seem to be returned without angle information by LAStools, so fill those up (theoretically, the pulse density raster does not capture all filled pixels, only last returns, but that should give a pretty good picture anyways)
              
              angle99th = round(as.numeric(global(scanangle,quantile,probs = 0.99,na.rm = T)),2)
              
              writeRaster(scanangle, filename = file.path(path_output,paste0("scanangle_abs",addendum_name,".tif")), overwrite = T)
              
              if (params_general$cleanup == T) file.remove(files_scanangle)
              
            }
            , error = function(e){
              return("WARNING! Scan angle could not be obtained\n")
            }
            
          )
          
          if (params_general$cleanup == T) {
            file.remove(files_pulsedensity_firstlast)
            file.remove(files_pulsedensity_scanangle)
          }
          
          #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
          cat("\nDTM creation with las2dem / blast2dem\n")
          #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
          
          # create default dtm
          step_processing = paste0("create DTMs")
          
          time_dtm_lasdef = las2dem(params_general = params_general, path_output = "", name_raster = "dtm_lasdef", kill = 200, step = resolution, type_output = "tif", option = "dtm")
          summary_full$time_dtm_lasdef = time_dtm_lasdef
          
          path_dtm_lasdef = file.path(params_general$path_data,"dtm_lasdef")
          files_dtm_lasdef = list.files.nonzero(path = path_dtm_lasdef, pattern = ".tif", full.names = TRUE)
          dtm_lasdef = vrt(files_dtm_lasdef)
          terra::crs(dtm_lasdef) = crs_scan
          
          if(ext(pulsedensity_first) != ext(dtm_lasdef)){
            cat("Warning! DTM extent is off and will be adjusted!\n")
            dtm_lasdef = crop(extend(dtm_lasdef, pulsedensity_first), pulsedensity_first)
          }
          
          writeRaster(dtm_lasdef, filename = file.path(path_output,paste0("dtm_lasdef",addendum_name,".tif")), overwrite = T)
          
          # create refined dtm, which will become the actual dtm
          # this should be identical to dtm_lasdef in most cases
          # time_dtm_refined = las2dem(params_general = params_general, path_output = "", name_raster = "dtm_refined", kill = 200, step = resolution, type_output = "tif", option = "dtm_refined")
          
          time_dtm = make.dtm_nooverhangs(params_general = params_general, path_output = "", path_products = path_output, name_raster = "dtm", kill = 200, step = resolution, type_output = "tif",classes_ground = "2 8",threshold_drop = 10)
          summary_full$time_dtm = time_dtm
          
          path_dtm = file.path(path_output,"dtm.tif")
          
          files_dtm = list.files.nonzero(path = path_dtm, pattern = ".tif", full.names = TRUE)
          dtm = vrt(files_dtm_lasdef) # ???? WHY files_dtm_lasdef ????
          terra::crs(dtm) = crs_scan
          
          cat("\nWrite dtm masks\n")
          # now define pulse density percentages with respect to dtm
          perc_pd02 = 100 * max(0, as.numeric(global(dtm,"notNA") - global(mask_pd02,"notNA")))/as.numeric(global(dtm,"notNA"))
          perc_pd04 = 100 * max(0, as.numeric(global(dtm,"notNA") - global(mask_pd04,"notNA")))/as.numeric(global(dtm,"notNA"))
          
          # Correct dtm dimensions
          dtm <- crop(dtm, dtm_lasdef)
          dtm_lasdef <- crop(dtm_lasdef, dtm)
          
          # now write out rasters depending on whether there is actually any improvement
          diff_refinement = dtm - dtm_lasdef
          mat_circular = terra::focalMat(diff_refinement, 10, type = "circle") # draw a 10 circle around each point
          mat_circular[mat_circular != 0] = 1
          diff_refinement_smoothed = terra::focal(diff_refinement, w=mat_circular, fun="mean", na.rm=T)
          mask_steep = ifel(diff_refinement_smoothed > -0.5 & diff_refinement_smoothed < 0.5, 1, NA)
          writeRaster(mask_steep, filename = file.path(path_output,paste0("mask_steep",addendum_name,".tif")), overwrite = T)
          
          cat("\nWrite dtm masks done!\n")
          
          # get percentage of steep areas
          perc_steep = 100 * max(0,as.numeric(global(diff_refinement_smoothed,"notNA") - global(mask_steep,"notNA")))/as.numeric(global(diff_refinement_smoothed,"notNA"))
          
          # read in again, not as vrt
          dtm = rast(file.path(path_output,paste0("dtm",addendum_name,".tif")))
          
          # create a mask
          outline_localCRS = clamp(dtm, lower = 1, upper = 1, values = T)
          outline_localCRS = as.polygons(outline_localCRS, values = F)
          outline_localCRS = fillHoles(outline_localCRS)
          
          # new in v.48
          # use standard algorithms to create specific dtm styles (wilderness/nature)
          #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
          cat("\nCreating additional DTM layers\n")
          #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
          # new in v.48
          
          cat("\nCreating highest DTM layers\n")
          # also add a layer with the basic ground points
          make.dtm_highest(params_general = params_general,path_output = "", name_raster = "dtm_highest", type_output = "tif", step = resolution, subcircle = 1, classes = "2 8")
          path_dtm_highest = file.path(params_general$path_data,"dtm_highest")
          files_dtm_highest = list.files.nonzero(path = path_dtm_highest, pattern = ".tif", full.names = TRUE)
          dtm_highest = vrt(files_dtm_highest); terra::crs(dtm_highest) = crs_scan
          if(ext(pulsedensity_first) != ext(dtm_highest)){
            cat("Warning! DTM extent is off and will be adjusted!\n")
            dtm_highest = crop(extend(dtm_highest, pulsedensity_first), pulsedensity_first)
          }
          
          writeRaster(dtm_highest, filename = file.path(path_output,paste0("dtm_highest",addendum_name,".tif")), overwrite = T)
          
          cat(paste0("Saved ", file.path(path_output,paste0("dtm_highest",addendum_name,".tif"))))
          cat("\nhighest DTM layers OK\n")
          
          # now create a mask for NA values in the DTM
          minradius_NA = 15
          dtm_nona_agg = aggregate(ifel(is.na(dtm_highest),NA,1), fact = 5, fun = "mean", na.rm = T)
          mat_circular = focalMat(dtm_nona_agg, minradius_NA, type = "circle") # draw a circle around each point
          mat_circular[mat_circular != 0] = 1
          dtm_nona_agg = terra::focal(dtm_nona_agg, w = mat_circular, fun = "mean", na.rm = T)
          mask_ground = terra::focal(ifel(is.na(dtm_nona_agg),1,NA), w = mat_circular, fun = "mean", na.rm = T)
          mask_noground = ifel(is.na(mask_ground), 1, NA)
          mask_noground = terra::resample(mask_noground, dtm_highest)
          writeRaster(mask_noground, filename = file.path(path_output,paste0("mask_noground",addendum_name,".tif")), overwrite = T)
          
          # get percentage of areas with ground points
          perc_nogrd = 100 * max(0,as.numeric(global(dtm,"notNA")-global(mask_noground,"notNA")))/as.numeric(global(dtm,"notNA"))
          
          # Common settings
          # Default: step is 25 m, sub is 5, bulge is 2 m, spike is 1+1 m, and offset is 0.05 m
          # Nature: step is 5 m, sub is 3, bulge is 1 m, spike is 1+1 m, and offset is 0.05 m
          # extra_fine changes sub to 7, bulge should be 1/10th of step size, but is clamped to between 1 and 2 by default
          
          dtms_custom = data.table(name = c("dtm_lasfine"), arguments_additional = c("-step 10 -bulge 1.0 -hyper_fine"))
          
          for(i in 1:nrow(dtms_custom)){
            name_dtm_custom = dtms_custom[i]$name
            arguments_additional = dtms_custom[i]$arguments_additional
            
            cat("Creating",name_dtm_custom,"\n")
            time_dtm_custom = make.dtm_custom(params_general = params_general, path_output = file.path(params_general$path_data,name_dtm_custom), name_raster = name_dtm_custom, kill = 200, step = resolution, arguments_additional = arguments_additional)
            name_time = paste0("time_",name_dtm_custom)
            summary_full[, (name_time) := time_dtm_custom]
            
            files_dtm_custom = list.files.nonzero(path = file.path(params_general$path_data,name_dtm_custom), pattern = ".tif", full.names = TRUE)
            dtm_custom = vrt(files_dtm_custom); terra::crs(dtm_custom) = crs_scan
            if(ext(pulsedensity_firstlast) != ext(dtm_custom)){
              cat("Warning! DTM extent is off and will be adjusted!\n")
              dtm_custom = crop(extend(dtm_custom,pulsedensity_firstlast),pulsedensity_firstlast)
            }
            writeRaster(dtm_custom, filename = file.path(path_output,paste0(name_dtm_custom,addendum_name,".tif")), overwrite = T)
            
            files_chunks_lasfine = list.files.nonzero(path = file.path(params_general$path_data,name_dtm_custom), pattern = ".laz", full.names = TRUE)
            files_chunk_lasfine_lax = list.files.nonzero(path = file.path(params_general$path_data,name_dtm_custom), pattern = ".lax", full.names = TRUE)
            
            if (params_general$cleanup == T) {
              file.remove(files_chunks_lasfine)
              file.remove(files_chunk_lasfine_lax)
            }
          }
          
          # to create an index of dtm uncertainty, we use the difference between dtm_lasfine and dtm_lasdef
          file_dtm_lasfine = file.path(path_output,paste0("dtm_lasfine",addendum_name,".tif"))
          
          perc_undtm = as.numeric(NA)
          
          if(file.exists(file_dtm_lasfine)){
            #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
            cat("\nCreate mask of DTM uncertainty \n")
            #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
            
            file_dtm_lasdef = file.path(path_output, paste0("dtm_lasdef",addendum_name,".tif"))
            if(!file.exists(file_dtm_lasdef)){
              file_dtm_lasdef = file.path(path_output, paste0("dtm",addendum_name,".tif"))
            }
            
            dtm_lasdef = rast(file_dtm_lasdef)
            dtm_lasfine = rast(file_dtm_lasfine)
            
            mat_circular = focalMat(dtm_lasdef, 10, type = "circle") # draw a 10 circle around each point
            mat_circular[mat_circular != 0] = 1
            dtm_lasdef_smoothed = terra::focal(dtm_lasdef, w = mat_circular, fun = "mean", na.rm = T)
            dtm_lasfine_smoothed = terra::focal(dtm_lasfine, w = mat_circular, fun = "mean", na.rm = T)
            
            diff_dtms = dtm_lasfine_smoothed - dtm_lasdef_smoothed
            
            mask_unstabledtm = ifel(diff_dtms < 2 & diff_dtms > -2,1,NA)
            writeRaster(mask_unstabledtm, filename = file.path(path_output,paste0("mask_unstabledtm",addendum_name,".tif")), overwrite = T)
            
            perc_undtm = 100 * max(0, as.numeric(global(diff_dtms,"notNA") - global(mask_unstabledtm,"notNA")))/as.numeric(global(diff_dtms,"notNA"))
          }
          
          if (params_general$cleanup == T) file.remove(files_dtm_lasdef)
          if (params_general$cleanup == T) file.remove(files_dtm)
          
          #%%%%%%%%%%%%%%%%%%%%%%#
          cat("\nDSM creation \n")
          #%%%%%%%%%%%%%%%%%%%%%%#
          
          step_processing = paste0("create DSMs")
          
          #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
          cat("\nCreating highest point-based DSM \n")
          #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
          
          time_dsm_highest = make.dsm_highest(params_general = params_general, path_output = file.path(params_general$path_data,"dsm_highest"), name_raster = "dsm_highest", type_output = "tif", step = resolution, subcircle = 0.1)
          
          summary_full$time_dsm_highest = time_dsm_highest
          
          files_dsm_highest = list.files.nonzero(path = file.path(params_general$path_data,"dsm_highest"), pattern = ".tif", full.names = TRUE)
          dsm_highest = vrt(files_dsm_highest); terra::crs(dsm_highest) = crs_scan
          if(ext(dsm_highest) != ext(dtm)){
            cat("Warning! DSM extent is off and will be adjusted!\n")
            dsm_highest = crop(extend(dsm_highest,dtm),dtm)
          }
          writeRaster(dsm_highest, filename = file.path(path_output,paste0("dsm_highest",addendum_name,".tif")), overwrite = T)
          
          chm_highest = dsm_highest - dtm; terra::crs(chm_highest) = crs_scan
          chm_highest = clamp(chm_highest, lower = 0, values = T)
          writeRaster(chm_highest, filename = file.path(path_output,paste0("chm_highest",addendum_name,".tif")), overwrite = T)
          
          # calculate some sumstats
          chm_mean = as.numeric(global(chm_highest, "mean",na.rm = T))
          chm_sd = as.numeric(global(chm_highest, "sd",na.rm = T))
          chm_perc99 = as.numeric(global(chm_highest, quantile, probs = 0.99, na.rm = T))
          chm_max = as.numeric(global(chm_highest, "max",na.rm = T))
          
          chm_min2 = ifel(chm_highest >= 2, 1, 0)
          chm_min10 = ifel(chm_highest >= 10, 1, 0)
          cc2 = as.numeric(global(chm_min2, "sum",na.rm = T))/as.numeric(global(chm_min2, "notNA"))
          cc10 = as.numeric(global(chm_min10, "sum",na.rm = T))/as.numeric(global(chm_min10, "notNA"))
          
          # we add a highnoise / cloud mask
          cat("Some CHM values are close to cutoff. These are potentially high noise / clouds that have not been removed\n")
          invmask_cloud = clamp(chm_highest, lower = height_lim - 1, values = F)
          invmask_cloud = clamp(invmask_cloud, upper = 1, lower = 1, values = T)
          invmask_cloud = terra::focal(invmask_cloud, w = 11, fun = "median", na.rm = T)
          mask_cloud = ifel(is.na(invmask_cloud),1,NA)
          writeRaster(mask_cloud, filename = file.path(path_output,paste0("mask_cloud",addendum_name,".tif")), overwrite = T)
          
          # get percentage of cloudy areas
          perc_cloud = 100 * max(0, as.numeric(global(dtm,"notNA") - global(mask_cloud,"notNA")))/as.numeric(global(dtm,"notNA"))
          
          if("tin" %in% types_dsm){
            #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
            cat("\nCreating TIN-based DSM \n")
            #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
            time_dsm_tin = las2dem(params_general = params_general, path_output = file.path(params_general$path_data,"dsm_tin"), name_raster = "dsm_tin", kill = 200, step = resolution, type_output = "tif", option = "dsm")
            summary_full$time_dsm_tin = time_dsm_tin
            
            files_dsm_tin = list.files.nonzero(path = file.path(params_general$path_data,"dsm_tin"), pattern = ".tif", full.names = TRUE)
            dsm_tin = vrt(files_dsm_tin); terra::crs(dsm_tin) = crs_scan
            if(ext(dsm_tin) != ext(dtm)){
              cat("Warning! DSM extent is off and will be adjusted!\n")
              dsm_tin = crop(extend(dsm_tin,dtm),dtm)
            }
            
            writeRaster(dsm_tin, filename = file.path(path_output,paste0("dsm_tin",addendum_name,".tif")), overwrite = T)
            
            chm_tin = dsm_tin - dtm; terra::crs(chm_tin) = crs_scan
            chm_tin = clamp(chm_tin, lower = 0, values = T)
            writeRaster(chm_tin, filename = file.path(path_output,paste0("chm_tin",addendum_name,".tif")), overwrite = T)
          }
          
          if("spikefree" %in% types_dsm)
          {
            #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
            cat("\nCreating Spikefree-based DSM \n")
            #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
            time_dsm_spikefree = las2dem(params_general = params_general, path_output = file.path(params_general$path_data,"dsm_spikefree"), name_raster = "dsm_spikefree", kill = 200, step = resolution, type_output = "tif", option = "dsm_spikefree")
            summary_full$time_dsm_spikefree = time_dsm_spikefree
            
            files_dsm_spikefree = list.files.nonzero(path = file.path(params_general$path_data,"dsm_spikefree"), pattern = ".tif", full.names = TRUE)
            dsm_spikefree = vrt(files_dsm_spikefree); terra::crs(dsm_spikefree) = crs_scan
            if(ext(dsm_spikefree) != ext(dtm)){
              cat("Warning! DSM extent is off and will be adjusted!\n")
              dsm_spikefree = crop(extend(dsm_spikefree,dtm),dtm)
            }
            
            writeRaster(dsm_spikefree, filename = file.path(path_output,paste0("dsm_spikefree",addendum_name,".tif")), overwrite = T)
            
            chm_spikefree = dsm_spikefree - dtm; terra::crs(chm_spikefree) = crs_scan
            chm_spikefree = clamp(chm_spikefree, lower = 0, values = T)
            writeRaster(chm_spikefree, filename = file.path(path_output,paste0("chm_spikefree",addendum_name,".tif")), overwrite = T)
          }
          
          if("lspikefree" %in% types_dsm & nrow(params_dsmadaptive) > 0){
            
            
            cat("LSPIKEFREE !")
            for(id_param in 1:nrow(params_dsmadaptive)){
              params_dsmadaptive_current = params_dsmadaptive[id_param]
              name_dsm_lspikefree = as.character(NA)
              if(nrow(params_dsmadaptive) == 1){
                name_dsm_lspikefree = "dsm_lspikefree"
              } else {
                name_dsm_lspikefree = paste0("dsm_lspikefree_multi",
                                             params_dsmadaptive_current$multi,
                                             "_slope",
                                             params_dsmadaptive_current$slope,
                                             "_offset",
                                             params_dsmadaptive_current$offset)
              }
              cat("Creating",name_dsm_lspikefree,"\n")
              
              time_dsm_lspikefree = make.dsm_locallyadaptive(params_general = params_general, 
                                                             path_output = file.path(params_general$path_data,name_dsm_lspikefree), 
                                                             name_raster = name_dsm_lspikefree, 
                                                             kill = 200, 
                                                             step = resolution, 
                                                             option = "spikefree", 
                                                             params_dsmadaptive = params_dsmadaptive_current, 
                                                             normalize = F)
              
              name_time = paste0("time_", name_dsm_lspikefree)
              summary_full[, (name_time) := time_dsm_lspikefree]
              
              files_dsm_lspikefree = list.files.nonzero(path = file.path(params_general$path_data, name_dsm_lspikefree), 
                                                        pattern = ".tif", 
                                                        full.names = TRUE)
              
              
              dsm_lspikefree = vrt(files_dsm_lspikefree); terra::crs(dsm_lspikefree) = crs_scan
              
              # # TO CHECK BEFORE CRASH ????
              # res(dsm_lspikefree); res(dtm)
              # origin(dsm_lspikefree); origin(dtm)
              # ext(dsm_lspikefree); ext(dtm)
              
              if(ext(dsm_lspikefree) != ext(dtm)){
                cat("Warning! DSM extent is off and will be adjusted!\n")
                dsm_lspikefree = crop(extend(dsm_lspikefree,dtm),dtm)
              }
              
              writeRaster(dsm_lspikefree, 
                          filename = file.path(path_output, paste0(name_dsm_lspikefree, addendum_name,".tif")), 
                          overwrite = T)
              
              
              # #TO RESTART AFTER LSPIKEFREE BUG...
              # dsm_lspikefree = terra::rast("/home/karl/work/dev/ALS_processing_GCA/03_processed_Luki2025/dsm_lspikefree.tiff")
              # dtm = terra::rast("/home/karl/work/dev/ALS_processing_GCA/03_processed_Luki2025/dtm.tiff")
              # chm_lspikefree = dsm_lspikefree - dtm
              # 
              # terra::crs(chm_lspikefree) = terra::crs(dsm_lspikefree)
              # chm_lspikefree = terra::clamp(chm_lspikefree, lower = 0, values = T)
              # terra::writeRaster(chm_lspikefree,
              #             filename = file.path("/home/karl/work/dev/ALS_processing_GCA/03_processed_Luki2025/chm_lspikefree.tif"),
              #             overwrite = T)
              
              chm_lspikefree = dsm_lspikefree - dtm; terra::crs(chm_lspikefree) = crs_scan
              chm_lspikefree = terra::clamp(chm_lspikefree, lower = 0, values = T)
              writeRaster(chm_lspikefree, 
                          filename = file.path(path_output,paste0(gsub("dsm","chm",name_dsm_lspikefree),"", addendum_name,".tif")), 
                          overwrite = T)
              
            }
          }
          
          if("lspikefree" %in% types_dsm & nrow(params_dsmadaptive) > 0 & !is.voidstring(params_general$path_lastools)){
            #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
            cat("\nCreating locally adaptive spikefree DSM \n")
            #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
            
            for(id_param in 1:nrow(params_dsmadaptive)){
              params_dsmadaptive_current = params_dsmadaptive[id_param]
              
              # simple name, in case only one parameter configuration is provided
              name_dsm_lspikefree = as.character(NA)
              if(nrow(params_dsmadaptive) == 1){
                name_dsm_lspikefree = "dsm_lspikefree"
              } else {
                name_dsm_lspikefree = paste0("dsm_lspikefree_multi",params_dsmadaptive_current$multi,"_slope",params_dsmadaptive_current$slope,"_offset",params_dsmadaptive_current$offset)
              }
              
              cat("Creating",name_dsm_lspikefree,"\n")
              time_dsm_lspikefree = make.dsm_locallyadaptive(params_general = params_general, path_output = file.path(params_general$path_data,name_dsm_lspikefree), name_raster = name_dsm_lspikefree, kill = 200, step = resolution, option = "spikefree", params_dsmadaptive = params_dsmadaptive_current, normalize = F, perturbation_max = perturbation_max, timeout_lspikefree_max = timeout_lspikefree_max)
              
              name_time = paste0("time_",name_dsm_lspikefree)
              summary_full[, (name_time) := time_dsm_lspikefree]
              
              files_dsm_lspikefree = list.files.nonzero(path = file.path(params_general$path_data,name_dsm_lspikefree), pattern = ".tif", full.names = TRUE)
              dsm_lspikefree = vrt(files_dsm_lspikefree); terra::crs(dsm_lspikefree) = crs_scan
              if(ext(dsm_lspikefree) != ext(dtm)){
                cat("Warning! DSM extent is off and will be adjusted!\n")
                dsm_lspikefree = crop(extend(dsm_lspikefree,dtm),dtm)
              }
              
              writeRaster(dsm_lspikefree, filename = file.path(path_output,paste0(name_dsm_lspikefree,addendum_name,".tif")), overwrite = T)
              
              if (params_general$cleanup == T) file.remove(files_dsm_lspikefree)
              
              chm_lspikefree = dsm_lspikefree - dtm; terra::crs(chm_lspikefree) = crs_scan
              chm_lspikefree = clamp(chm_lspikefree, lower = 0, values = T)
              writeRaster(chm_lspikefree, filename = file.path(path_output,paste0(gsub("dsm","chm",name_dsm_lspikefree),"",addendum_name,".tif")), overwrite = T)
              
              if(id_param == 1){
                files_perturbation = list.files.nonzero(path = file.path(params_general$path_data,name_dsm_lspikefree), pattern = "_perturbation.shp", full.names = TRUE)
                if(length(files_perturbation) > 0){
                  contours_perturbation = vect(lapply(files_perturbation, vect))
                  writeVector(contours_perturbation, filename = file.path(path_output,paste0("contours_perturbation",addendum_name,".shp")), overwrite = T)
                }
              }
            }
          }
          if (params_general$cleanup == T) file.remove(files_dsm_lspikefree)
          if (params_general$cleanup == T) file.remove(files_dsm_tin)
          if (params_general$cleanup == T) file.remove(files_dsm_highest)
          
          #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
          cat("\nCreate normalized point cloud \n")
          #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
          step_processing = paste0("normalize point cloud")
          cat("NOT DONE IN OPENSOURCE VERSION\n")
          # NOTE: we do not update params_general, the normalized point clouds are going to be stored in a subfolder, indicated by params_general_normalized$path_data
          params_general$cleanup = F
          params_general_normalized = NULL
          params_general_normalized = normalize.pointcloud(params_general = params_general, path_output = "", path_dtm = "", update.path = T)
          
          laserpenetration_mean = NULL
          
          if(estimate.laserpenetration == T){
            step_processing = paste0("estimate laser penetration")
            #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
            cat("\nEstimate laser penetration \n")
            #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
            # we estimate the laser's capability of penetrating vegetation
            # we do so by computing all first returns above a certain height threshold in the normalized point cloud (set by ignore.below), which should yield only vegetation returns
            # we then compute the number of first returns above that threshold that are also last returns (i.e., single returns)
            # penetration is 1 - the ratio between the two
            # reasoning: in a relatively open canopy, lots of laser pulses are going to hit the ground, and thus inflate the number of single returns
            
            get.laserpenetration(params_general_normalized, path_output = "", type_output = "tif", step = 25, ignore.below = 2)
            
            files_returns_first = list.files.nonzero(path = file.path(params_general_normalized$path_data,"laserpenetration"), pattern = "_first.tif", full.names = TRUE)
            files_returns_firstlast = list.files.nonzero(path = file.path(params_general_normalized$path_data,"laserpenetration"), pattern = "_firstlast.tif", full.names = TRUE)
            
            # using the virtual raster method from terra package
            returns_first = vrt(files_returns_first)
            terra::crs(returns_first) = crs_scan
            returns_firstlast = vrt(files_returns_firstlast)
            terra::crs(returns_firstlast) = crs_scan
            
            returns_first_sum = as.numeric(global(returns_first,"sum",na.rm = T))
            returns_firstlast_sum = as.numeric(global(returns_firstlast,"sum",na.rm = T))
            laserpenetration_mean = 1.0 - returns_firstlast_sum/returns_first_sum
            
            laserpenetration = ifel(returns_first > 0, 1.0 - returns_firstlast/returns_first, NA)
            writeRaster(laserpenetration, filename = file.path(path_output,paste0("laserpenetration",addendum_name,".tif")), overwrite = T)
            
            if (params_general$cleanup == T) file.remove(files_returns_first)
            if (params_general$cleanup == T) file.remove(files_returns_firstlast)
            
          }
          
          step_processing = paste0("create DSMs from normalized point cloud")
          
          if("highest_norm" %in% types_dsm){
            #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
            cat("\nCreating highest point-based CHM \n")
            #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
            time_chmn_highest = make.dsm_highest(params_general = params_general_normalized, path_output = file.path(params_general$path_data,"chmn_highest"), name_raster = "chmn_highest", type_output = "tif", step = resolution, subcircle = 0.1)
            
            summary_full$time_chmn_highest = time_chmn_highest
            
            files_chmn_highest = list.files.nonzero(path = file.path(params_general$path_data,"chmn_highest"), pattern = ".tif", full.names = TRUE)
            chmn_highest = vrt(files_chmn_highest); terra::crs(chmn_highest) = crs_scan
            chmn_highest = clamp(chmn_highest, lower = 0, values = T)
            if(ext(chmn_highest) != ext(dtm)){
              cat("Warning! CHM extent is off and will be adjusted!\n")
              chmn_highest = crop(extend(chmn_highest,dtm),dtm)
            }
            writeRaster(chmn_highest, filename = file.path(path_output,paste0("chmn_highest",addendum_name,".tif")), overwrite = T)
            if (params_general$cleanup == T) file.remove(files_chmn_highest)
          }
          
          if("tin_norm" %in% types_dsm){
            #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
            cat("\nCreating TIN-based CHM \n")
            #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
            time_chmn_tin = las2dem(params_general = params_general_normalized, path_output = file.path(params_general$path_data,"chmn_tin"), name_raster = "chmn_tin", kill = 200, step = resolution, type_output = "tif", option = "dsm")
            summary_full$time_chmn_tin = time_chmn_tin
            
            files_chmn_tin = list.files.nonzero(path = file.path(params_general$path_data,"chmn_tin"), pattern = ".tif", full.names = TRUE)
            chmn_tin = vrt(files_chmn_tin); terra::crs(chmn_tin) = crs_scan
            chmn_tin = clamp(chmn_tin, lower = 0, values = T)
            if(ext(chmn_tin) != ext(dtm)){
              cat("Warning! CHM extent is off and will be adjusted!\n")
              chmn_tin = crop(extend(chmn_tin,dtm),dtm)
            }
            writeRaster(chmn_tin, filename = file.path(path_output,paste0("chmn_tin",addendum_name,".tif")), overwrite = T)
            if (params_general$cleanup == T) file.remove(files_chmn_tin)
            
          }
          
          if("lspikefree_norm" %in% types_dsm & nrow(params_dsmadaptive) > 0){
            #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
            cat("\nCreating locally adaptive spikefree CHM \n")
            #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
            
            for(id_param in 1:nrow(params_dsmadaptive)){
              params_dsmadaptive_current = params_dsmadaptive[id_param]
              
              # simple name, in case only one parameter configuration is provided
              if(nrow(params_dsmadaptive) == 1){
                name_chmn_lspikefree = "chmn_lspikefree"
              } else {
                name_chmn_lspikefree = paste0("chm_lspikefree_multi",params_dsmadaptive_current$multi,"_slope",params_dsmadaptive_current$slope,"_offset",params_dsmadaptive_current$offset)
              }
              
              cat("Creating",name_chmn_lspikefree,"\n")
              time_chmn_lspikefree = make.dsm_locallyadaptive(params_general = params_general_normalized, path_output = file.path(params_general$path_data, name_chmn_lspikefree), name_raster = name_chmn_lspikefree, kill = 200, step = resolution, option = "spikefree", params_dsmadaptive = params_dsmadaptive_current, normalize = F, perturbation_max = perturbation_max, timeout_lspikefree_max = timeout_lspikefree_max)
              
              name_time = paste0("time_",name_chmn_lspikefree)
              summary_full[, (name_time) := time_chmn_lspikefree]
              
              files_chmn_lspikefree = list.files.nonzero(path = file.path(params_general$path_data,name_chmn_lspikefree), pattern = ".tif", full.names = TRUE)
              chmn_lspikefree = vrt(files_chmn_lspikefree); terra::crs(chmn_lspikefree) = crs_scan
              chmn_lspikefree = clamp(chmn_lspikefree, lower = 0, values = T)
              if(ext(chmn_lspikefree) != ext(dtm)){
                cat("Warning! CHM extent is off and will be adjusted!\n")
                chmn_lspikefree = crop(extend(chmn_lspikefree,dtm),dtm)
              }
              writeRaster(chmn_lspikefree, filename = file.path(path_output,paste0(name_chmn_lspikefree,addendum_name,".tif")), overwrite = T)
              if (params_general$cleanup == T) file.remove(files_chmn_lspikefree)
              
            }
          }
          
          if(!is.null(resolution_sumstatspc)){
            step_processing = paste0("create point cloud statistics")
            #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
            cat("\nCompute canopy summary statistics from point cloud\n")
            #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
            for(resolution_current in resolution_sumstatspc){
              cat("Using resolution: ",resolution_current,"\n")
              dir_sumstats = file.path(params_general$path_data, paste0("sumstats",resolution_current,"_pc"))
              compute.sumstats_pc(params_general = params_general_normalized, path_output = dir_sumstats, resolution = resolution_current, cutoff_height = 0, type_output = "tif")
              
              files_sumstats = data.table(file = list.files(dir_sumstats,pattern = ".tif", full.names = T))
              files_sumstats[, file_basename := gsub(".tif","",basename(file))]
              files_sumstats[, name := substr(file_basename, nchar(file_basename)-2, nchar(file_basename))]
              
              sumstats_pc = rast(files_sumstats$file)
              names(sumstats_pc) = files_sumstats$name
              
              cat("Write sumstats to file\n")
              writeRaster(sumstats_pc, filename = file.path(path_output,paste0("sumstats",resolution_current,"_pc",addendum_name,".tif")), overwrite = T)
              if (params_general$cleanup == T) file.remove(files_sumstats)
              
            }
          }
          
          if(path_output_laznorm != ""){
            step_processing = paste0("write normalized laz files")
            if(!dir.exists(path_output_laznorm)) dir.create(path_output_laznorm)
            
            files_laznorm = list.files(params_general_normalized$path_data, pattern = ".laz", full.names = T)
            if(length(files_laznorm) > 0){
              #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
              cat("\nSave normalized laz files \n")
              #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
              file.copy(files_laznorm, path_output_laznorm)
            }
          }
          
          time_end_current = Sys.time()
          
          #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
          cat("\nProduce combined mask \n")
          #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
          step_processing = paste0("combine masks")
          files_masks = list.files(path_output, pattern = "mask_")
          files_masks = files_masks[files_masks %like% "mask_pd02" | files_masks %like% "mask_cloud" | files_masks %like% "mask_noground"]
          masks_combined = terra::rast(file.path(path_output,files_masks))
          mask_combined = sum(masks_combined)
          terra::writeRaster(mask_combined, filename = file.path(path_output,paste0("mask_combined",addendum_name,".tif")), overwrite = T)
          
          #%%%%%%%%%%%%%%%%%%%%%%%%%%%#
          cat("\nProduce synthesis \n")
          #%%%%%%%%%%%%%%%%%%%%%%%%%%%#
          
          step_processing = paste0("produce synthesis")
          # get dates
          # the global encoding bit is often not set correctly, but common values that indicate adjusted GPS standard time should be 1,5,17 and 21
          # in practice it is easier to just check whether the timestamps fall within the week and sort out the rest manually (i.e., overwrite by setting is.stdtime = T/F)
          # we set a minimum timestamp of -369280000 (01/01/2000), everything below is suspect for a lidar scan
          dates_acquisition = data.table(mindt = as.Date(NA), maxdt = as.Date(NA))
          use.stdtime = F
          if(is.na(is.stdtime)){
            if(!is.na(summary_full$mingps) & !is.na(summary_full$maxgps)){
              if(summary_full$mingps > -369280000 & summary_full$maxgps > -369280000 & summary_full$mingps != 0 & !(summary_full$mingps >= 0 & summary_full$mingps <= 604800 & summary_full$maxgps >= 0 & summary_full$maxgps <= 604800)){
                use.stdtime = T
              }
            }
          } else {
            if(is.stdtime == T){
              use.stdtime = T
            }
          }
          
          if(use.stdtime == T){
            # conversion cf. https://groups.google.com/g/lastools/c/dU8CWeVrhNE
            # JAN6_1980 = 315964800; #Seconds between Jan 1, 1970 (POSIX time zero) and Jan 6, 1980 (GPS time zero)
            GPS_OFFSET = 1e9
            dates_acquisition = data.table(
              mindt = as.Date(as.POSIXct(summary_full$mingps + GPS_OFFSET, origin = "1980-01-06")),
              maxdt = as.Date(as.POSIXct(summary_full$maxgps + GPS_OFFSET, origin = "1980-01-06"))
            )
          } else {
            dates_from_path = get.dates_from_path(path_input)
            if(!is.na(dates_from_path$mindt) & !is.na(dates_from_path$maxdt)){
              dates_acquisition = dates_from_path
            } else {
              dates_acquisition$mindt = as.Date(metadata$acq_mindt,"%d/%m/%Y")
              dates_acquisition$maxdt = as.Date(metadata$acq_maxdt,"%d/%m/%Y")
            }
          }
          
          cat("\ncreate a polygon + metadata\n")
          # create a polygon + metadata
          outline_localCRS$dataset = metadata$dataset
          outline_localCRS$site = metadata$site
          outline_localCRS$source = metadata$source
          outline_localCRS$citation = metadata$citation
          outline_localCRS$contact = metadata$contact
          outline_localCRS$access = metadata$access
          outline_localCRS$license = metadata$license
          outline_localCRS$updated = metadata$updated
          outline_localCRS$id_orig = metadata$id_orig
          outline_localCRS$acq_type = metadata$acq_type
          outline_localCRS$acq_crs = metadata$acq_crs
          outline_localCRS$acq_units = metadata$acq_units
          outline_localCRS$acq_mindt = metadata$acq_mindt
          outline_localCRS$acq_maxdt = metadata$acq_maxdt
          outline_localCRS$acq_system = metadata$acq_system
          outline_localCRS$acq_AGL = metadata$acq_AGL
          outline_localCRS$acq_swath = metadata$acq_swath
          outline_localCRS$acq_wavel = metadata$acq_wavel
          outline_localCRS$acq_fpsize = metadata$acq_fpsize
          outline_localCRS$acq_lasdiv = metadata$acq_lasdiv
          outline_localCRS$notes = metadata$notes
          outline_localCRS$issues = metadata$issues
          outline_localCRS$height_lim = height_lim
          outline_localCRS$angle_lim = ifelse(is.null(angle_lim),as.numeric(NA),angle_lim)
          outline_localCRS$class_rm = class_rm
          outline_localCRS$exclass_rm = exclass_rm
          
          cat("\nadd additional parameters\n")
          # add additional parameters
          outline_localCRS$prms_lspkf = paste0(params_dsmadaptive, collapse = " ")
          
          cat("\nadd computed scan information\n")
          # add computed scan information
          description_crs = terra::crs(crs_scan, describe = T) # simplify crs to name / EPSG code if possible
          terra::crs(outline_localCRS) <- crs_scan
          outline_localCRS$crs = ifelse(is.na(description_crs$name) | is.na(description_crs$code) | is.na(description_crs$authority), 
                                        crs_scan, 
                                        paste0(description_crs$name," (",paste0(description_crs[,c("authority","code")],collapse = ":"),")"))
          outline_localCRS$lon = xmin(centroids(project(outline_localCRS,"EPSG:4326")))
          outline_localCRS$lat = ymin(centroids(project(outline_localCRS,"EPSG:4326")))
          outline_localCRS$area_km2 = round(expanse(project(outline_localCRS,"EPSG:4326"))/1000000,3)
          outline_localCRS$GPSadjstd = summary_full$encoding_global
          outline_localCRS$mingps = round(summary_full$mingps,3)
          outline_localCRS$maxgps = round(summary_full$maxgps,3)
          outline_localCRS$mindt = as.character(format(dates_acquisition$mindt,"%d/%m/%Y"))
          outline_localCRS$maxdt = as.character(format(dates_acquisition$maxdt,"%d/%m/%Y"))
          outline_localCRS$pd_point = round(summary_full$density_points_mean,3)
          outline_localCRS$pd_pulse = round(summary_full$density_pulses_mean,3)
          outline_localCRS$sdpd_pulse = round(summary_full$density_pulses_sd,3)
          outline_localCRS$frac_grnd = round(summary_full$fraction_ground_mean,3)
          outline_localCRS$angle99th = angle99th
          
          cat("\ninformation from masks\n")
          # information from masks
          outline_localCRS$perc_pd02 = round(perc_pd02,2)
          outline_localCRS$perc_pd04 = round(perc_pd04,2)
          outline_localCRS$perc_steep = round(perc_steep,2)
          outline_localCRS$perc_nogrd = round(perc_nogrd,2)
          outline_localCRS$perc_undtm = round(perc_undtm,2)
          outline_localCRS$perc_cloud = round(perc_cloud,2)
          
          cat("\nadd additional information about the resulting landscape\n")
          # add additional information about the resulting landscape
          outline_localCRS$elev = round(as.numeric(global(dtm,"mean",na.rm = T)))
          outline_localCRS$elev_min = round(as.numeric(global(dtm,"min",na.rm = T)))
          outline_localCRS$elev_max = round(as.numeric(global(dtm,"max",na.rm = T)))
          outline_localCRS$chm_mean = round(chm_mean,2)
          outline_localCRS$chm_sd = round(chm_sd,2)
          outline_localCRS$chm_perc99 = round(chm_perc99,2)
          outline_localCRS$chm_max = round(chm_max,2)
          outline_localCRS$cc2 = round(cc2,2)
          outline_localCRS$cc10 = round(cc10,2)
          
          cat("\nadd information about the processing\n")
          # add information about the processing
          outline_localCRS$type_os = type_os
          outline_localCRS$type_arch = type_architecture
          outline_localCRS$v_lastools = get.version_lastools(params_general)
          
          cat("\na little bit of path magic\n")
          # a little bit of path magic
          path_origin = path_input
          if(grepl(pattern = paste0(".",type_file),x = path_input, fixed = TRUE) & file_test("-f",path_input)){ # file test may be overkill here
            path_origin = dirname(path_input)
          }
          wd_processing = getwd()
          setwd(path_origin); path_absolute = getwd(); setwd(wd_processing)
          if(grepl(pattern = paste0(".",type_file),x = path_input, fixed = TRUE) & file_test("-f",path_input)){ # file test may be overkill here
            path_absolute = file.path(path_absolute, basename(path_input))
          }
          outline_localCRS$dir_input = path_absolute
          setwd(path_output); path_absolute = getwd(); setwd(wd_processing)
          outline_localCRS$dir_output = path_absolute
          
          cat("\nother information\n")
          # other information
          outline_localCRS$time_start = as.character(format(time_start_current,"%d/%m/%Y %Hh%M"))
          outline_localCRS$time_end = as.character(format(time_end_current,"%d/%m/%Y %Hh%M"))
          outline_localCRS$mins_total = round(as.numeric(difftime(time_end_current,time_start_current,units = "mins")),2)
          outline_localCRS$mins_grdcl = round(as.numeric(summary_full$time_ground_basic),2)
          outline_localCRS$mins_grdrf = round(as.numeric(summary_full$time_ground_refinement),2)
          outline_localCRS$mins_lspkf = round(ifelse(any(colnames(summary_full) == "time_dsm_lspikefree"),as.numeric(summary_full[,c("time_dsm_lspikefree")]), as.numeric(NA)),2)
          outline_localCRS$type_file = type_file
          outline_localCRS$n_files = n_files
          outline_localCRS$n_cores = params_general$n_cores
          outline_localCRS$adjacents = ifelse(retile == "include.adjacents","included","ignored")
          outline_localCRS$size_MBin = size_MB_input
          
          cat("\ncalculate size\n")
          # calculate size
          files_processed = list.files(path_output, recursive = TRUE, full.names = T)
          size_byte_output = sum(file.info(files_processed)$size)
          size_MB_output = round(size_byte_output/(1024*1024),2)
          outline_localCRS$size_MBout = size_MB_output
          
          cat("\nwrite out the local CRS outline\n")
          # write out the local CRS outline
          writeVector(outline_localCRS, filename = file.path(path_output,paste0("outline_localCRS",addendum_name,".shp")), overwrite = T)
          
          cat("\nconvert to WGS84\n")
          # convert to WGS84
          outline_WGS84 = project(outline_localCRS, "EPSG:4326")
          writeVector(outline_WGS84, filename = file.path(path_output,paste0("outline_WGS84",addendum_name,".shp")), overwrite = T)
          pause.xseconds(5) # pause to not overload system with requests (sometimes folders might be locked through synchronization with Dropbox/OneDrive, etc.)
          fwrite(as.data.table(outline_WGS84), file = file.path(path_output,paste0("summary_processing",addendum_name,".csv")))
          
          #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
          cat("\nWrapping up. Writing _INFO_ files\n")
          #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
          # since v50, summary files are not written out anymore; instead, we provide _INFO_ files
          # pause to not overload system with requests (sometimes folders might be locked through synchronization with Dropbox/OneDrive, etc.)
          pause.xseconds(5)
          make.info(path_output = path_output, name_file = "_INFO_.txt",params_general = params_general)
          pause.xseconds(5)
          make.info_products(path_output = path_output, name_file = "_INFO_products.csv")
          pause.xseconds(5)
          make.info_summary_processing(path_output = path_output, name_file = "_INFO_summary_processing.csv")
          
          # fwrite(summary_bytile, file = file.path(path_output,paste0("summary_bytile",addendum_name,".csv")))
          # fwrite(summary_full, file = file.path(path_output,paste0("summary_full",addendum_name,".csv")))
        } else {
          # since v50, summary files are not written out anymore; instead, we provide _INFO_ files
          # pause to not overload system with requests (sometimes folders might be locked through synchronization with Dropbox/OneDrive, etc.)
          pause.xseconds(5)
          make.info(path_output = path_output, name_file = "_INFO_.txt", params_general = params_general)
          pause.xseconds(5)
          make.info_products(path_output = path_output, name_file = "_INFO_products.csv")
          pause.xseconds(5)
          make.info_summary_processing(path_output = path_output, name_file = "_INFO_summary_processing.csv")
          # summary_full = data.table(n_files = n_files)
          # fwrite(summary_full, file = file.path(path_output,paste0("summary_full",addendum_name,".csv")))
        }
        
        if(length(clusters_data) > 1) unlink(clusters_data[[i]]$path_data, recursive = T)
        
        
      }, error = function(e){
        nberrors = nberrors + 1
        
        error_message = paste(
          "Error at step:", step_processing, "\n",
          "Message:", conditionMessage(e), "\n",
          "Timestamp:", Sys.time(), "\n"
        )
        
        # Write to file
        writeLines(error_message, con = file.path(path_output,"_ERROR.txt"), sep = "\n")
      })
    }
    
    if(nberrors > 0){
      return(paste0(nberrors, " subsets with errors (check _ERROR.txt files)"))
    } else {
      return("processed")
    }
  }
}

# this is the high-level function that is called on an entire data set
# it looks for subdirectories that contain las files and then processes them via process.datasubset

# parameters
# - dir_dataset: where the data are
# - dir_processed: where the processed data will be stored
# - type_file: las/laz
# - tmpdir_processing: the directory for processing
# - name_job: name of current job (in case data set is processed several times, use v1, v2, or similar)

# TODO: merge dir_structure with information_processing (lots of information is duplicated between the two) to simplify script
process.dataset = function(name_job, type_file, dir_dataset, dir_processed, tmpdir_processing, path_lastools, metadata = NULL, resolution = 1, n_cores = 4, size_tile = 500, size_buffer = 25, retile = T, cleanup = T, nbclusters_forced = NULL, force.utm = F, remove.buffer = F, remove.vlr = F, remove.evlr = F, factor_rescale = NULL, force.recompute = F, path_output_lazclean = "", path_output_laznorm = "", types_dsm = c("tin","spikefree", "lspikefree"), params_dsmadaptive = data.table(multi = 3.1, slope = 1.75, offset = 2.1), resolution_sumstatspc = NULL, add.timestamp = F, estimate.laserpenetration = F, type_os = "automatic", type_architecture = "64", by_file = F, perturbation_max = 0.1, timeout_lspikefree_max = 600, overwrite.crs = F, use.blast2dem = F, is.stdtime = NA, height_lim = 125, angle_lim = NULL, class_rm = c(), exclass_rm = c(), force.type_point = NULL, logfile = "", patterns_skip = c(), print.summary_job = F){
  
  if(logfile != "" & dir.exists(dirname(logfile))){
    # new in v.50: create a log file for the most common issues
    logfile_dt = data.table(path = character(), issue = character())
    fwrite(logfile_dt, file = logfile, append = F)
  } else {
    logfile = ""
  }
  
  # we remove any trailing slashes from end of paths
  dir_dataset = gsub("/$", "", dir_dataset)
  dir_processed = gsub("/$", "", dir_processed)
  tmpdir_processing = gsub("/$", "", tmpdir_processing)
  path_lastools = gsub("/$", "", path_lastools)
  
  # then we normalize the paths (this prevents path length problems on windows)
  dir_dataset = normalizePath(dir_dataset)
  dir_processed = normalizePath(dir_processed)
  tmpdir_processing = normalizePath(tmpdir_processing)
  #path_lastools = normalizePath(path_lastools)
  
  # obtain basic information
  dir_structure = list.files(path = dir_dataset, recursive = TRUE, pattern = paste0(".", type_file))
  
  # check LAStools
  has.path_lastools = T #check.path_lastools(path_lastools = path_lastools, logfile)
  
  if(length(dir_structure) == 0){
    # Test existence of files
    cat("Did not find any files of type",type_file,"in dataset directory:",dir_dataset,"\n")
    cat("Did you specificy the correct directory and the correct file type? (should be one of .laz, .las, .LAS)")
  } else if(has.path_lastools) {
    # check user input
    #check.license(path_lastools, logfile)
    type_os = check.type_os(type_os, logfile)
    type_architecture = check.type_architecture(type_architecture, type_os, logfile)
    n_cores = check.n_cores(n_cores, type_os, logfile)
    # use.blast2dem = update.blast2dem(use.blast2dem, type_os)
    # cat("\n")
    
    # new in v.42: processing  by file
    if(by_file == F){
      dir_structure = unique(dirname(dir_structure))
    }
    dir_structure.dt = data.table(dir_structure)
    dir_structure.dt[, dir_structure_clean := ifelse(like(dir_structure, ".zip",fixed = T) | like(dir_structure, paste0(".",type_file),fixed = T), substr(dir_structure,1,nchar(dir_structure)-4), dir_structure)]
    dir_structure.dt[, path_input := file.path(dir_dataset,dir_structure)]
    dir_structure.dt[, path_output := file.path(dir_processed, dir_structure_clean)]
    dir_structure.dt[, name_acquisition := basename(dir_structure_clean)]
    
    # create output file with information on processing progress
    information_processing = data.table(id = 1:nrow(dir_structure.dt), name_job = name_job, name_acquisition = dir_structure.dt$name_acquisition, path_input = dir_structure.dt$path_input, path_output = dir_structure.dt$path_output, time_start = as.character(NA), time_end = as.character(NA), time_processing_minutes = as.character(NA), status = as.character(NA))
    
    # restart
    # information_processing = fread(file = file.path(dir_processed,paste0(name_job,"_",time_start_job,".csv")))
    # information_processing$time_start = as.character(information_processing$time_start)
    # information_processing$time_end = as.character(information_processing$time_end)
    
    # View(information_processing)
    # i = 1
    
    time_start_job = as.character(format(Sys.time(), "%Y-%m-%d_%Hh%M")) # simplified format
    
    for(i in 1:nrow(dir_structure.dt)){
      # now replicate the structure for the results
      time_start = Sys.time()
      dir_current = dir_structure.dt[i]
      
      if(!dir.exists(dir_current$path_output)){dir.create(dir_current$path_output, recursive = TRUE)}
      
      # updated in v43 to avoid overwriting of already processed data in cases where data can be split into multiple subsets
      # updated in v.1.0.0 replacing summary_full with _INFO_summary_processing (the last file to be created)
      file_summary_current = list.files(dir_current$path_output,pattern = "_INFO_summary_processing", recursive = T)
      dirs_current = list.dirs(dir_current$path_output, recursive = F)
      
      # updated in v.1.0.0 Pattern matching to subset what should (not) be processed
      skip.pattern = F
      if(length(patterns_skip) > 0){
        for(pattern_skip in patterns_skip){
          skip.pattern = ifelse(like(dir_current$path_input,pattern_skip, fixed = T), T, skip.pattern)
        }
      }
      
      if(!(length(file_summary_current) == 0 | length(file_summary_current) < length(dirs_current) | force.recompute == T)){
        cat(i,"| Skipping:", dir_current$path_input," (already processed) | Time:", as.character(format(time_start,"%d/%m/%Y %Hh%M")),"\n")
        information_processing[i]$status = "already processed"
      } else if(skip.pattern == T){
        cat(i,"| Skipping:", dir_current$path_input," (path matches patterns_skipped) | Time:", as.character(format(time_start,"%d/%m/%Y %Hh%M")),"\n")
        information_processing[i]$status = "path matches patterns_skipped"
      } else {
        cat(i,"| Processing:", dir_current$path_input," | Time:", as.character(format(time_start,"%d/%m/%Y %Hh%M")),"\n")
        # remove old files
        dirs_output_current = list.dirs(dir_current$path_output, full.names = T)
        dirs_output_current = dirs_output_current[-1]
        if(length(dirs_output_current) > 0) unlink(dirs_output_current, recursive = T)
        
        files_output_current = list.files(dir_current$path_output, full.names = T)
        # if(length(files_output_current) > 0) file.remove(files_output_current)
        
        checkpoint_step = 0
        
        # expected_output_products = c("dtm_supplied.tif",
        #                              "scanangle_abs.tif",
        #                              "mask_unstabledtm.tif",
        #                              "chm_spikefree.tif")
        # 
        # check_output_files_for_reload = function(files_output_current, expected_output_products, reload=F){
        #   if(reload == F) {
        #     file.remove(files_output_current)
        #     checkpoint_step = 0}
        #   else if(length(files_output_current) == 0){
        #     cat("no output files\n")
        #     checkpoint_step = 0}
        #   else{
        #     matched_files = expected_output_products[expected_output_products %in% basename(files_output_current)]
        #     if (length(matched_files) == length(expected_output_products)){
        #       cat("enough ! all files are already here")}
        #     else{
        #       cat("Found last checkpoint step: ", tail(matched_files, n = 1), "\n")
        #     }
        #     checkpoint_step = length(matched_files)
        #   }
        #   return (checkpoint_step)
        # }
        # checkpoint_step = check_output_files_for_reload(files_output_current, expected_output_products, reload=T)
        
        # start the processing
        path_lastools = path_lastools
        path_tmp = tmpdir_processing
        path_input = dir_current$path_input
        path_output = dir_current$path_output
        
        addendum_name = ""
        if(add.timestamp == T){
          addendum_name = as.character(format(time_start,"%d/%m/%Y %Hh%M"))
        }
        
        deduplicate = T
        denoise = T
        reclassify = T
        # path_output_laznorm = ""
        # size_buffer = ifelse(dir_current$name_acquisition %like% "pd_0.5", 100, ifelse(dir_current$name_acquisition %like% "pd_1", 50, size_buffer)) # forced extension of buffering for specific pulse density thinned point clouds # TODO: remove
        cat("Size buffer:",size_buffer,"\n")
        
        status = tryCatch(
          process.datasubset(path_lastools = path_lastools, path_tmp = path_tmp, path_input = path_input, path_output = path_output, type_file = type_file, addendum_name = addendum_name, metadata = metadata, retile = retile,deduplicate = deduplicate, denoise = denoise, reclassify = reclassify, resolution = resolution, height_lim = height_lim, angle_lim = angle_lim, n_cores = n_cores, size_tile = size_tile, size_buffer = size_buffer, cleanup = cleanup, nbclusters_forced = nbclusters_forced, force.utm = force.utm, remove.buffer = remove.buffer, remove.vlr = remove.vlr, remove.evlr = remove.evlr, factor_rescale = factor_rescale, path_output_lazclean = path_output_lazclean, path_output_laznorm = path_output_laznorm, types_dsm = types_dsm, params_dsmadaptive = params_dsmadaptive, resolution_sumstatspc = resolution_sumstatspc, estimate.laserpenetration = estimate.laserpenetration, type_os = type_os, type_architecture = type_architecture, perturbation_max = perturbation_max, timeout_lspikefree_max = timeout_lspikefree_max, overwrite.crs = overwrite.crs, use.blast2dem = use.blast2dem, is.stdtime = is.stdtime, class_rm = class_rm, exclass_rm = exclass_rm, force.type_point = force.type_point, logfile = logfile, checkpoint_step = checkpoint_step),
          error = function(e){
            if(file.exists(logfile)){
              logfile_dt = data.table(path = path_output, issue = paste0("ERROR! Processing failed with message:", conditionMessage(e)))
              fwrite(logfile_dt, file = logfile, append = T)
            }
            
            # Optionally, print or return the error message
            message("Error captured and written to file: ", conditionMessage(e))
            NULL  # Return NULL or an appropriate fallback
          }
        )
        
        information_processing[i]$status = status
      } 
      
      time_end = Sys.time()
      time_processing = difftime(time_end,time_start,units = "mins")
      
      # append to the general output files
      information_processing[i]$time_start = as.character(format(time_start,"%d/%m/%Y %Hh%M"))
      information_processing[i]$time_end = as.character(format(time_end,"%d/%m/%Y %Hh%M"))
      information_processing[i]$time_processing_minutes = as.character(round(time_processing))
      
      # v.1.0.0 remove default printing of job summary. Since jobs can be run at different levels, this might be confusing. Summary information is included for each geo-unit anyways
      if(print.summary_job == T){
        # add a one second pause (otherwise Windows sometimes complains about writing to a file that is not properly closed yet from the previous iteration)
        Sys.sleep(1)
        fwrite(information_processing, file = file.path(dir_processed,paste0(name_job,"_",time_start_job,".csv")), sep = "\t",col.names = T)
      }
    }
    # new in v.1.0.0: return processing summary to console at the end
    return(information_processing)
  }
}

#%%%%%%%%%%%%%%%%%%%%%%%%%%%#
#### 5. Checking results ####
#%%%%%%%%%%%%%%%%%%%%%%%%%%%#

# custom function to find paths for subsets of a processed lidar data set
# if data set was not split, just provides the path to the main directory of the processed data
read.locations_subsets = function(path_locations){
  locations = vect(path_locations)
  locations = aggregate(locations,"cluster")
  if(length(unique(locations$cluster)) > 1){
    locations$path_processed = paste0(dirname(path_locations),"/part_",locations$cluster)
  } else {
    locations$path_processed = dirname(path_locations)
  }
  return(locations)
}
# function to rapidly assess processing results
# plots 4 rasters: pulse density, dtm, chm_highest, chm_pitfree
# dir_processed: directory where processed data are stored, can be an entire 03_processed folder, or a subfolder (as long as subfolder contains shapefiles from the processing, i.e. lidar scans split into "parts" do not work)
# idx_location: if several sites were processed, provide indexes of sites to be plotted
# remove_buffer: shapefiles from processing come with a typical buffer of 25m, which can be removed; the removed fraction can also be extended to remove artifacts at the edges
# dirsave_graphs: a directory to save graphs in, if NULL, visualization within R/Rstudio
# dirsave_locations: a directory to save the (transformed/aggregated) shapefiles in
# ext_crop: a terra extent object, either with absolute coordinates, or relative coordinates (when between 0 and 1 both in x and y direction)
# addendum_name: for saving to file, add a custom name

visualize.processing = function(dir_processed, idx_location = NULL, remove_buffer = 25, dirsave_graphs = NULL, dirsave_locations = NULL, ext_crop = NULL, addendum_name = ""){
  
  dir_processed = gsub("/$", "", dir_processed) # remove trailing slashes
  
  #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
  cat("Reading in and combining all delineated scans and subsets of scans\n")
  #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
  outlines_input = list.files(dir_processed, pattern = "outlines_input.shp", full.names = T, recursive = T)
  outlines_input = lapply(outlines_input, read.locations_subsets)
  outlines_input = vect(outlines_input)
  outlines_input$path_processed = gsub(dir_processed,"",outlines_input$path_processed, fixed = T)
  outlines_input$path_processed = sapply(outlines_input$path_processed,function(x){if(substring(x,1,1) == "/"){x = substring(x,2,)};return(x)})
  outlines_input$id_location = 1:nrow(outlines_input)
  noutlines_input = nrow(outlines_input)
  
  cat("Found",noutlines_input,"distinctive and separately processed locations\n")
  
  if(!is.null(dirsave_locations)) writeVector(outlines_input, filename = file.path(dirsave_locations,"outlines_input.shp"), overwrite = T)
  
  # define visualization: either plot one specific location or a range of locations; the default is to plot all of them
  locations_toplot = idx_location
  if(is.null(locations_toplot)) locations_toplot = 1:noutlines_input
  if(addendum_name != "") addendum_name = paste0("",addendum_name)
  
  for(i in locations_toplot){
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
    cat("Visualize location",i,"\n")
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
    
    location_current = outlines_input[i]
    path_full = unique(file.path(dir_processed,location_current$path_processed))
    
    # now read in rasters and crop, if necessary
    file_pd = list.files(path_full, pattern = "pulsedensity", full.name = T)
    file_pd = file_pd[!file_pd %like% "scanangle" & !file_pd %like% "agg"]
    file_pd = file_pd[1]
    pd = crop(rast(file_pd),location_current)
    
    # crop to custom extent
    ext_crop_current = copy(ext_crop)
    if(!is.null(ext_crop_current)){
      # create custom extent from relative extent (between 0 and 1 in both dimensions)
      if(range(ext_crop_current)[1] <= 1 & range(ext_crop_current)[2] <= 1){
        # get properties of current extent
        ext_pd = ext(pd)
        xmin_pd = xmin(pd)
        ymin_pd = ymin(pd)
        xrange_pd = range(ext_pd)[1]
        yrange_pd = range(ext_pd)[2]
        
        # convert into cropped coordinates
        xmin_crop = xmin_pd + xrange_pd * xmin(ext_crop_current)
        xmax_crop = xmin_pd + xrange_pd * xmax(ext_crop_current)
        ymin_crop = ymin_pd + yrange_pd * ymin(ext_crop_current)
        ymax_crop = ymin_pd + yrange_pd * ymax(ext_crop_current)
        ext_crop_current = ext(xmin_crop, xmax_crop, ymin_crop, ymax_crop)
      }
      pd = crop(pd, ext_crop_current)
    }
    
    file_dtm = list.files(path_full, pattern = "dtm", full.name = T)[1]
    dtm = crop(rast(file_dtm),pd)
    
    file_chm_highest = list.files(path_full, pattern = "chm_highest", full.name = T)[1]
    chm_highest = crop(rast(file_chm_highest),pd)
    
    file_chm_lspikefree = list.files(path_full, pattern = "chm_lspikefree", full.name = T)[1]
    chm_lspikefree = crop(rast(file_chm_lspikefree),pd)
    
    # check if plot should be written to file and, if so, format it
    ratio_dim_yx = dim(pd)[1]/dim(pd)[2]
    if(!is.null(dirsave_graphs)){
      if(!dir.exists(dirsave_graphs)) dir.create(dirsave_graphs, recursive = T)
      width_pdf = NULL
      height_pdf = NULL
      
      # determine output ratios of the pdf depending on the plot shape and size
      if(ratio_dim_yx > 2){
        height_pdf = 20
        width_pdf = height_pdf / ratio_dim_yx * 4
      } else if(ratio_dim_yx > 1){
        height_pdf = 20
        width_pdf = height_pdf / ratio_dim_yx
      } else if(ratio_dimxy > 0.5){
        width_pdf = 20
        height_pdf = width_pdf * ratio_dim_yx
      } else {
        width_pdf = 20
        height_pdf = width_pdf * ratio_dim_yx * 4
      }
      pdf(file = file.path(dirsave_graphs,paste0(i,"_overview",addendum_name,".pdf")), width = width_pdf, height = height_pdf)
    } else {
      if(length(locations_toplot) > 1 & i > 1) readline(prompt="Press [enter] to continue")
      par_mfrow_previous = par()$mfrow
    }
    
    # actual plotting
    if(ratio_dim_yx > 2){
      par(mfrow = c(1,4))
    } else if(ratio_dim_yx < 0.5){
      par(mfrow = c(4,1))
    } else {
      par(mfrow = c(2,2))
    }
    
    # default plotting
    plot(pd, col = viridis(100, option = "plasma"), main = "Pulse density (1/m2)", cex.main = 1.5)
    par(usr = c(0, 1, 0, 1))
    legend(0.1,0.9, legend = paste0("#",i,": ",location_current$path_processed), bty = "o", bg = "white", text.col = "black", cex= 0.65, xjust = 0, yjust = 1)
    plot(dtm, col = viridis(100, option = "turbo"), main = "DTM (m)", cex.main = 1.5)
    plot(chm_highest, col = viridis(100), main = "CHM, highest return (m)", cex.main = 1.5)
    plot(chm_lspikefree, col = viridis(100), main = "CHM, local spikefree (m)", cex.main = 1.5)
    
    # reset
    if(!is.null(dirsave_graphs)){
      dev.off()
    } else {
      par(mfrow = par_mfrow_previous)
    }
  }
}

compare.raster = function(file_rst1, file_rst2, factor_extent = 0.1){
  rst1 = trim(rast(file_rst1))
  rst1 = crop(rst1, ext(rst1) * factor_extent)
  rst2 = trim(rast(file_rst2))
  rst2 = crop(rst2, rst1)
  rst1 = crop(rst1, rst2)
  
  diff = rst2 - rst1
  
  min_rst1 = as.numeric(global(rst1, "min", na.rm = T))
  min_rst2 = as.numeric(global(rst2, "min", na.rm = T))
  max_rst1 = as.numeric(global(rst1, "max", na.rm = T))
  max_rst2 = as.numeric(global(rst2, "max", na.rm = T))
  range_fixed = c(min(c(min_rst1,min_rst2)),max(c(max_rst1,max_rst2)))
  
  par(mfrow = c(2,2))
  
  plot(rst1, col = viridis(100), range = range_fixed, main = "Raster 1")
  plot(rst2, col = viridis(100), range = range_fixed, main = "Raster 2")
  plot(diff, col = viridis(100, option = "turbo"), main = "Differences")
  hist(diff, main = "Histogram of differences")
  
  par(mfrow = c(1,1))
  
  return(data.table(rmse = as.numeric(sqrt(global(diff*diff,"mean",na.rm = T))), bias = as.numeric(global(diff,"mean",na.rm = T))))
}

