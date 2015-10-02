using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MySql.Data.MySqlClient;
using MySql.Data.Types;
using Emgu.CV;
using Emgu.CV.Structure;
using Emgu.Util;
using Emgu.CV.UI;
using Emgu.CV.CvEnum;
using Emgu.CV.Features2D;
using Emgu.CV.GPU;
using System.IO;
using System.Drawing;
using System.ComponentModel;

namespace Iris_Recognition
{

    public class IrisDB
    {
        private MySqlConnection connection;
        private string server;
        private string database;
        private string uid;
        private string password;

        public IrisDB()
        {
            // Host system
            server = "localhost";

            // db name = irisdb
            database = "irisdb";

            // mysql uid
            uid = "root";

            //mysql password
            password = "ac";

            string connectionString;

            connectionString = "SERVER=" + server + ";" + "DATABASE=" +
            database + ";" + "UID=" + uid + ";" + "PASSWORD=" + password + ";";

            //open connection to the mysql
            connection = new MySqlConnection(connectionString);
        }

        //open connection to database
        private bool OpenConnection()
        {
            try
            {
                connection.Open();
                return true;
            }
            catch (MySqlException ex)
            {
                return false;
            }
        }

        //Close connection
        private bool CloseConnection()
        {
            try
            {
                connection.Close();
                return true;
            }
            catch (MySqlException ex)
            {
                return false;
            }
        }

        //Insert statement
        public void Insert(IrisDBEntry entry)
        {
            //open connection
            if (this.OpenConnection() == true)
            {
                //create command and assign the query and connection from the constructor

                MySqlCommand cmd = connection.CreateCommand();

                //insert statement
                cmd.CommandText = "INSERT INTO iris VALUES(@id,@name,@InputImage,@IrisCode)";
                MemoryStream ms = new MemoryStream();
                //insert each value
                cmd.Parameters.AddWithValue("@id", entry.id.ToString());
                cmd.Parameters.AddWithValue("@name", entry.name);
                entry.InputImage.Bitmap.Save(ms, System.Drawing.Imaging.ImageFormat.Bmp);
                cmd.Parameters.AddWithValue("@InputImage", ms.GetBuffer());
                ms = new MemoryStream();
                entry.IrisCode.Bitmap.Save(ms, System.Drawing.Imaging.ImageFormat.Bmp);
                cmd.Parameters.AddWithValue("@IrisCode", ms.GetBuffer());
                //Execute command
                cmd.ExecuteNonQuery();

                //close connection
                this.CloseConnection();
            }
        }

        //Update statement
        public void Update()
        {
        }

        //Delete statement
        public void Delete()
        {
        }

        //Select statement
        public List<IrisDBEntry> Select(int start = 0, int limit = 6)
        {
            List<IrisDBEntry> Entries = new List<IrisDBEntry>();
            //Open connection
            if (this.OpenConnection() == true)
            {
                //Create Command
                MySqlCommand cmd = connection.CreateCommand();
                cmd.CommandText = "SELECT * FROM iris LIMIT @start,@limit";
                cmd.Parameters.AddWithValue("@start", start);
                cmd.Parameters.AddWithValue("@limit", limit);

                //Create a data reader and Execute the command
                MySqlDataReader dataReader = cmd.ExecuteReader();

                //Read the data and store them in the list
                while (dataReader.Read())
                {
                    IrisDBEntry entry = new IrisDBEntry();
                    entry.id = Guid.Parse((string)dataReader[0]);
                    entry.name = (string)dataReader[1];
                     byte[] array = (byte[])dataReader[2];
                    TypeConverter tc = TypeDescriptor.GetConverter(typeof(Bitmap));
                    Bitmap image = (Bitmap)tc.ConvertFrom(array);
                    entry.InputImage = new Image<Gray, byte>(image);
                    array = (byte[])dataReader[3];
                    image = (Bitmap)tc.ConvertFrom(array);
                    entry.IrisCode = new Image<Gray, byte>(image);
                    Entries.Add(entry);
                }

                //close Data Reader
                dataReader.Close();

                //close Connection
                this.CloseConnection();
            }
            return Entries;
        }

        //Count statement
        public int Count()
        {

            string query = "SELECT Count(*) FROM iris";
            int Count = -1;

            //Open Connection
            if (this.OpenConnection() == true)
            {
                //Create Mysql Command
                MySqlCommand cmd = new MySqlCommand(query, connection);

                //ExecuteScalar will return one value
                Count = int.Parse(cmd.ExecuteScalar() + "");

                //close Connection
                this.CloseConnection();

                return Count;
            }
            else
            {
                return Count;
            }
        }

    }
}
