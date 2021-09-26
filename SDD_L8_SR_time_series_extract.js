// Running in Google Earth Engine
// High-resolution SDD mapping
// Write by Yuan He
// 2021-08

// First, load World lake raster datasets, water occurrancy images and country raster datasets in GEE.

// Function to cloud mask from the pixel_qa band of Landsat 8 SR data.
function maskL8sr(image) {
	// Bits 3 and 5 are cloud shadow and cloud, respectively.
	var cloudShadowBitMask = 1 << 3;
	var cloudsBitMask = 1 << 5;
	var qa = image.select('pixel_qa');
	var mask = qa.bitwiseAnd(cloudShadowBitMask).eq(0)
	    .and(qa.bitwiseAnd(cloudsBitMask).eq(0));
	return image.updateMask(mask).divide(10000)
	    .select("B[0-9]*")
	    .copyProperties(image, ["system:time_start"]);
      }
      
// Iteration of L8 SR
var startYear = 2014;
var endYear = 2020;
// Create a list of years by generating a sequence from start and end years 
var years = ee.List.sequence(startYear, endYear);
// Generate annual summed image mosaics 
var imgCollection_l8 = ee.ImageCollection.fromImages(
years.map(function (year) {
	var startDate = ee.Date.fromYMD(year, 8, 1).format("YYYY-MM-dd");
	var endDate = ee.Date.fromYMD(year, 10, 31).format("YYYY-MM-dd");
	var base = ee.ImageCollection("LANDSAT/LC08/C01/T1_SR")
	.filterDate(startDate, endDate)
	var annual = ee.ImageCollection("LANDSAT/LC08/C01/T1_SR")
	.filterDate(startDate, endDate)
	.map(maskL8sr)
	.select("B1","B2","B3","B4","B5","B7")
	.median();
	return annual.multiply(water)
	.set('year', year)
	.copyProperties(base.first(),["system:time_start"])
}));

// Merge L5/7/8 collection
var imgCollection = imgCollection_l8.mean();
print("imgCollection", imgCollection);

var imgCollection_time = imgCollection_l8
print("imgCollection_time", imgCollection_time);

// export geotiff to asset
Export.image.toAsset({
image: imgCollection,
description: "Asset-30m-landsat-bands",
assetId: "Global_landsat8_SR_6bands",
scale: 30,
maxPixels:1e13
})

// Reduce the region. The region parameter is the Feature geometry.
var ft = ee.FeatureCollection(ee.List([]))

function batch_export(image, ini){
var date = image.get('year');
var Lake_Collection = World_lake.map(function lake_compute(feature){
	var name = feature.get('Lake_name');
	var lon = feature.get('Pour_long');
	var lat = feature.get('Pour_lat');
	var meanDictionary = image.reduceRegion({
	reducer: ee.Reducer.mean(),
	geometry: feature.geometry().bounds(),
	scale: 30,
	maxPixels: 1e13,
	tileScale: 16
}); 
	return ee.Feature(null, {'Name': name,
				'Year': date,
				'Long': lon,
				'Lat': lat,
				'B1': meanDictionary.get('B1'),
				'B2': meanDictionary.get('B2'),
				'B3': meanDictionary.get('B3'),
				'B4': meanDictionary.get('B4'),
				'B5': meanDictionary.get('B5'),
				'B7': meanDictionary.get('B7')})
})
// print(Lake_Collection);
var inift = ee.FeatureCollection(ini);
return inift.merge(Lake_Collection)
}

var test2 = ee.FeatureCollection(imgCollection_l8.iterate(batch_export, ft));
print("test2",test2.limit(2));

// Export a .csv table of date
Export.table.toDrive({
collection: test2,
description: 'global_10more_time-v4-year',
folder: 'GEE_KD_out',
fileFormat: 'CSV',
})