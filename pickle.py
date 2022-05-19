    def load_properties(self, case_identifier):
        with open(os.path.join(self.output_folder, "%s.pkl" % case_identifier), 'rb') as f:
            properties = pickle.load(f)
        return properties

    def save_properties(self, case_identifier, properties):
        with open(os.path.join(self.output_folder, "%s.pkl" % case_identifier), 'wb') as f:
            pickle.dump(properties, f)

            
   with open(os.path.join(output_folder_stage, "%s.pkl" % case_identifier), 'wb') as f:
      pickle.dump(properties, f)
        
  all_classes = load_pickle(join(input_folder_with_cropped_npz, 'dataset_properties.pkl'))['all_classes']
