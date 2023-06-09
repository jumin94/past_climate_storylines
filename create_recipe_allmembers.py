import glob
import os
import yaml
rootpath="/shera/datos/CMIP/CMIP6"

short_name = "ua"
mip = "Amon"
file_list_Amon = glob.glob(os.path.join(rootpath,f"*/*/*/ssp585/*/{mip}/tas/*/*/*2015*.nc"))

dict_attr_list = [{'dataset':'nada'}]
dict_attr_list_Omon = []
for file in file_list_Amon:
    file_list_Omon = glob.glob(os.path.join(rootpath,f"*/*/*/ssp585/*/Omon/tos/*/*/*2015*.nc"))
    file_attrs = file.split("/")[-9:-2]
    keys = ["dataset","exp","ensemble","mip","short_name","grid"]
    file_list_Omon = glob.glob(os.path.join(rootpath+'/ScenarioMIP/'+('/').join(file_attrs[:4])+"/Omon/tos/*/*/*2015*.nc"))
    #print(rootpath+'/ScenarioMIP/'+('/').join(file_attrs[:4])+"/Omon/tos/*/*/*2014*.nc")
    #print(file_list_Omon)
    lista  = [m["dataset"] for m in dict_attr_list]
    if len(file_list_Omon) != 0:
        file_attrs_Omon = file_list_Omon[0].split("/")[-8:-2]
        dict_attr = {key: value for key,value in zip(keys,file_attrs[1:])}
        dict_attr["exp"] = ["historical", "ssp585"]
        dict_attr["project"] = "CMIP6"
        print(f"{file}")
        dict_attr.pop("mip", None)
        dict_attr.pop("short_name", None)
        dict_attr_list.append(dict_attr)
        dict_attr_Omon = {key: value for key,value in zip(keys,file_attrs_Omon)}
        dict_attr_Omon["exp"] = ["historical", "ssp585"]
        dict_attr_Omon["project"] = "CMIP6"
        print(f"{file}")
        dict_attr_Omon.pop("mip", None)
        dict_attr_Omon.pop("short_name", None)
        dict_attr_list_Omon.append(dict_attr_Omon)
    else:
        print('No existe tos para '+file_attrs[1])
    
list_Omon = [i for n, i in enumerate(dict_attr_list_Omon)
            if i not in dict_attr_list_Omon[n + 1:]]

list_Amon = [i for n, i in enumerate(dict_attr_list)
            if i not in dict_attr_list[n + 1:]]            

with open("/datos/julia.mindlin/past_climate_storylines/recipes/Amon_Omon_recipe_allmembers.yml", "w") as f:
    yaml.dump({"datasets_Amon": list_Amon,"datasets_Omon": list_Omon}, f, default_flow_style=False)

print(list_Amon.keys())
print('cantidad de miembros: ',len(list_Amon))
