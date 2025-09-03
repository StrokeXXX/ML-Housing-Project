resource "azurerm_machine_learning_compute_instance" "housing_compute" {
  name                          = "housing-compute"
  machine_learning_workspace_id = azurerm_machine_learning_workspace.main.id
  virtual_machine_size         = "Standard_DS3_v2"
}
