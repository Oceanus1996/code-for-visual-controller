using UnityEngine;
using UnityEditor;
using System.IO;
using System.Text;

public class SimpleObjectsExporter : EditorWindow
{
    [MenuItem("Tools/Export Objects Size & Position")]
    public static void ShowWindow()
    {
        GetWindow<SimpleObjectsExporter>("物体位置和大小导出");
    }

    void OnGUI()
    {
        if (GUILayout.Button("导出物体位置和大小"))
        {
            ExportObjectsSizeAndPosition();
        }
    }

    void ExportObjectsSizeAndPosition()
    {
        // 获取所有带有Renderer组件的对象
        Renderer[] renderers = FindObjectsOfType<Renderer>();
        
        StringBuilder csv = new StringBuilder();
        csv.AppendLine("名称,路径,位置X,位置Y,位置Z,大小X,大小Y,大小Z");
        
        int count = 0;
        foreach (Renderer renderer in renderers)
        {
            if (renderer.enabled && renderer.gameObject.activeInHierarchy)
            {
                string name = renderer.gameObject.name;
                string objectPath = GetGameObjectPath(renderer.gameObject);
                Vector3 position = renderer.transform.position;
                Vector3 size = renderer.bounds.size;
                
                csv.AppendLine($"\"{name}\",\"{objectPath}\",{position.x},{position.y},{position.z},{size.x},{size.y},{size.z}");
                count++;
            }
        }
        
        string filePath = EditorUtility.SaveFilePanel("保存物体位置和大小信息", "", "objects_size_position.csv", "csv");
        if (!string.IsNullOrEmpty(filePath))
        {
            File.WriteAllText(filePath, csv.ToString());
            Debug.Log($"已导出 {count} 个物体的位置和大小信息到: {filePath}");
        }
    }
    
    string GetGameObjectPath(GameObject obj)
    {
        string objPath = obj.name;
        Transform parent = obj.transform.parent;
        
        while (parent != null)
        {
            objPath = parent.name + "/" + objPath;
            parent = parent.parent;
        }
        
        return objPath;
    }
}